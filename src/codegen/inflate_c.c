
typedef struct { const int64_t *d; int64_t n, pos; uint32_t buf; int cnt; } jn_br;
typedef struct { uint16_t count[16]; uint16_t symbol[288]; } jn_huff;

static int jn_br_bit(jn_br *b) {
  if (!b->cnt) {
    if (b->pos >= b->n) return -1;
    b->buf = (uint32_t)(b->d[b->pos++] & 0xFF);
    b->cnt = 8;
  }
  int v = (int)(b->buf & 1u);
  b->buf >>= 1;
  b->cnt--;
  return v;
}

static int64_t jn_br_bits(jn_br *b, int n) {
  int64_t v = 0;
  for (int i = 0; i < n; i++) {
    int bit = jn_br_bit(b);
    if (bit < 0) return -1;
    v |= ((int64_t)bit) << i;
  }
  return v;
}

static int jn_huff_build(jn_huff *h, const uint8_t *len, int n) {
  memset(h->count, 0, sizeof(h->count));
  for (int i = 0; i < n; i++) h->count[len[i]]++;
  if (h->count[0] == n) return 0;
  int left = 1;
  for (int l = 1; l <= 15; l++) {
    left <<= 1;
    left -= h->count[l];
    if (left < 0) return -1;
  }
  uint16_t offs[16];
  offs[1] = 0;
  for (int l = 1; l < 15; l++) offs[l + 1] = (uint16_t)(offs[l] + h->count[l]);
  for (int i = 0; i < n; i++)
    if (len[i]) h->symbol[offs[len[i]]++] = (uint16_t)i;
  return left;
}

static int jn_huff_decode(jn_br *b, const jn_huff *h) {
  int code = 0, first = 0, index = 0;
  for (int len = 1; len <= 15; len++) {
    int bit = jn_br_bit(b);
    if (bit < 0) return -1;
    code |= bit;
    int cnt = h->count[len];
    if (code - first < cnt) return (int)h->symbol[index + (code - first)];
    index += cnt;
    first += cnt;
    first <<= 1;
    code <<= 1;
  }
  return -1;
}

static int jn_inflate(const int64_t *src, int64_t src_n, int64_t *out,
                      int64_t out_n) {
  static const uint16_t LB[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17,
    19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
  static const uint8_t LE[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
    2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
  static const uint16_t DB[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49,
    65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577};
  static const uint8_t DE[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
    6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
  static const uint8_t ORDER[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4,
    12, 3, 13, 2, 14, 1, 15};
  jn_br br;
  br.d = src; br.n = src_n; br.pos = 0; br.buf = 0; br.cnt = 0;
  int64_t len = 0;
  int last = 0;
  while (!last) {
    int bit = jn_br_bit(&br);
    if (bit < 0) return -1;
    last = bit;
    int64_t t = jn_br_bits(&br, 2);
    if (t < 0) return -1;
    if (t == 0) {
      br.cnt = 0;
      int64_t ln = jn_br_bits(&br, 16);
      int64_t nln = jn_br_bits(&br, 16);
      if (ln < 0 || nln < 0) return -1;
      if (((ln ^ 0xFFFF) & 0xFFFF) != nln) return -1;
      for (int64_t i = 0; i < ln; i++) {
        int64_t b = jn_br_bits(&br, 8);
        if (b < 0 || len >= out_n) return -1;
        out[len++] = b;
      }
    } else if (t == 1 || t == 2) {
      jn_huff lit, dist;
      if (t == 1) {
        uint8_t fl[288], fd[32];
        for (int i = 0; i < 144; i++) fl[i] = 8;
        for (int i = 144; i < 256; i++) fl[i] = 9;
        for (int i = 256; i < 280; i++) fl[i] = 7;
        for (int i = 280; i < 288; i++) fl[i] = 8;
        for (int i = 0; i < 32; i++) fd[i] = 5;
        if (jn_huff_build(&lit, fl, 288) < 0) return -1;
        if (jn_huff_build(&dist, fd, 32) < 0) return -1;
      } else {
        int64_t hlit = jn_br_bits(&br, 5);
        int64_t hdist = jn_br_bits(&br, 5);
        int64_t hclen = jn_br_bits(&br, 4);
        if (hlit < 0 || hdist < 0 || hclen < 0) return -1;
        hlit += 257; hdist += 1; hclen += 4;
        uint8_t cl[19];
        memset(cl, 0, sizeof(cl));
        for (int64_t i = 0; i < hclen; i++) {
          int64_t v = jn_br_bits(&br, 3);
          if (v < 0) return -1;
          cl[ORDER[i]] = (uint8_t)v;
        }
        jn_huff clh;
        if (jn_huff_build(&clh, cl, 19) < 0) return -1;
        uint8_t lens[320];
        int64_t nl = 0;
        while (nl < hlit + hdist) {
          int sym = jn_huff_decode(&br, &clh);
          if (sym < 0) return -1;
          if (sym < 16) {
            if (nl >= 320) return -1;
            lens[nl++] = (uint8_t)sym;
          } else if (sym == 16) {
            int64_t r = jn_br_bits(&br, 2);
            if (r < 0 || nl == 0) return -1;
            r += 3;
            if (nl + r > 320) return -1;
            uint8_t prev = lens[nl - 1];
            for (int64_t k = 0; k < r; k++) lens[nl++] = prev;
          } else if (sym == 17) {
            int64_t r = jn_br_bits(&br, 3);
            if (r < 0) return -1;
            r += 3;
            if (nl + r > 320) return -1;
            for (int64_t k = 0; k < r; k++) lens[nl++] = 0;
          } else if (sym == 18) {
            int64_t r = jn_br_bits(&br, 7);
            if (r < 0) return -1;
            r += 11;
            if (nl + r > 320) return -1;
            for (int64_t k = 0; k < r; k++) lens[nl++] = 0;
          } else return -1;
        }
        if (hlit > 288 || hdist > 32) return -1;
        if (jn_huff_build(&lit, lens, (int)hlit) < 0) return -1;
        if (jn_huff_build(&dist, lens + hlit, (int)hdist) < 0) return -1;
      }
      for (;;) {
        int sym = jn_huff_decode(&br, &lit);
        if (sym < 0) return -1;
        if (sym < 256) {
          if (len >= out_n) return -1;
          out[len++] = sym;
        } else if (sym == 256) {
          break;
        } else {
          int li = sym - 257;
          if (li >= 29) return -1;
          int64_t ex = jn_br_bits(&br, LE[li]);
          if (ex < 0) return -1;
          int64_t length = (int64_t)LB[li] + ex;
          int ds = jn_huff_decode(&br, &dist);
          if (ds < 0 || ds >= 30) return -1;
          int64_t dex = jn_br_bits(&br, DE[ds]);
          if (dex < 0) return -1;
          int64_t d = (int64_t)DB[ds] + dex;
          if (d > len || len + length > out_n) return -1;
          for (int64_t k = 0; k < length; k++) out[len] = out[len - d], len++;
        }
      }
    } else {
      return -1;
    }
    if (len > out_n) return -1;
  }
  return len == out_n ? 0 : -1;
}
