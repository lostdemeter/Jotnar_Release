//! Shared host scaffold for persistent GPU warm-frame workers.
//!
//! The CUDA and HIP emitters supply the op sequence and API names; this module
//! emits the config parser, shared-buffer mappings, handshake, and frame loop.

use crate::codegen::infer::CKind;

pub(crate) struct WarmBuffer {
    pub name: String,
    pub kind: CKind,
    pub elems: usize,
}

fn ctype(kind: CKind) -> &'static str {
    match kind {
        CKind::F => "float",
        CKind::I => "int64_t",
        CKind::T => "void",
    }
}

fn bytes(b: &WarmBuffer) -> usize {
    b.elems * if b.kind == CKind::I { 8 } else { 4 }
}

pub(crate) struct WarmApi<'a> {
    pub target: &'a str,
    pub malloc: &'a str,
    pub memcpy: &'a str,
    pub host_to_device: &'a str,
    pub device_to_host: &'a str,
    pub sync: &'a str,
}

pub(crate) fn emit_main(
    api: &WarmApi<'_>,
    inputs: &[WarmBuffer],
    intermediates: &[WarmBuffer],
    outputs: &[WarmBuffer],
    ops: &str,
) -> String {
    let mut s = String::new();
    s.push_str(
        "#include <sys/mman.h>\n#include <sys/stat.h>\n#include <fcntl.h>\n#include <unistd.h>\n",
    );
    s.push_str(
        "static void *__warm_map(const char *path, size_t bytes) {\n\
         \x20 int __fd = open(path, O_RDWR); if (__fd < 0) return NULL;\n\
         \x20 struct stat __st; if (fstat(__fd, &__st) != 0 || (size_t)__st.st_size < bytes) { close(__fd); return NULL; }\n\
         \x20 void *__p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, __fd, 0);\n\
         \x20 close(__fd);\n\
         \x20 return __p == MAP_FAILED ? NULL : __p;\n\
         }\n\n",
    );
    s.push_str("int warm_main(int argc, char **argv) {\n");
    s.push_str("  if (argc != 3) return 9;\n");
    s.push_str("  FILE *__cf = fopen(argv[2], \"r\"); if (!__cf) return 8;\n");
    s.push_str("  char __line[8192];\n");
    s.push_str("  if (!fgets(__line, sizeof __line, __cf)) return 8;\n");
    s.push_str("  if (strncmp(__line, \"jotnar-warm 1\", 13) != 0) return 8;\n");
    s.push_str("  if (!fgets(__line, sizeof __line, __cf)) return 8;\n");
    s.push_str("  char __target[32];\n");
    s.push_str(&format!(
        "  if (sscanf(__line, \"target %31s\", __target) != 1 || strcmp(__target, \"{}\") != 0) return 8;\n",
        api.target
    ));
    for i in 0..inputs.len() {
        s.push_str(&format!("  char __pin{i}[4096];\n"));
        s.push_str("  if (!fgets(__line, sizeof __line, __cf)) return 8;\n");
        s.push_str(&format!(
            "  if (sscanf(__line, \"input %4095s\", __pin{i}) != 1) return 8;\n"
        ));
    }
    for i in 0..outputs.len() {
        s.push_str(&format!("  char __pout{i}[4096];\n"));
        s.push_str("  if (!fgets(__line, sizeof __line, __cf)) return 8;\n");
        s.push_str(&format!(
            "  if (sscanf(__line, \"output %4095s\", __pout{i}) != 1) return 8;\n"
        ));
    }
    s.push_str("  fclose(__cf);\n");

    // One declaration/allocation per unique stream name.
    let mut all: Vec<&WarmBuffer> = Vec::new();
    for b in inputs
        .iter()
        .chain(intermediates.iter())
        .chain(outputs.iter())
    {
        if !all.iter().any(|x| x.name == b.name) {
            all.push(b);
        }
    }
    for b in &all {
        s.push_str(&format!("  {} *{} = NULL;\n", ctype(b.kind), b.name));
    }
    for b in &all {
        s.push_str(&format!(
            "  CK({}(&{}, {}));\n",
            api.malloc,
            b.name,
            bytes(b)
        ));
    }
    for (i, b) in inputs.iter().enumerate() {
        s.push_str(&format!(
            "  void *__hin{i} = __warm_map(__pin{i}, {}); if (!__hin{i}) return 8;\n",
            bytes(b)
        ));
    }
    for (i, b) in outputs.iter().enumerate() {
        s.push_str(&format!(
            "  void *__hout{i} = __warm_map(__pout{i}, {}); if (!__hout{i}) return 8;\n",
            bytes(b)
        ));
    }
    s.push_str("  unsigned char __ready = 'R';\n");
    s.push_str("  if (fwrite(&__ready, 1, 1, stdout) != 1) return 8;\n");
    s.push_str("  fflush(stdout);\n");
    s.push_str("  for (;;) {\n");
    s.push_str("    unsigned char __tok = 0;\n");
    s.push_str("    if (fread(&__tok, 1, 1, stdin) != 1) break;\n");
    s.push_str("    if (__tok == 0) break;\n");
    for (i, b) in inputs.iter().enumerate() {
        s.push_str(&format!(
            "    CK({}({}, __hin{i}, {}, {}));\n",
            api.memcpy,
            b.name,
            bytes(b),
            api.host_to_device
        ));
    }
    s.push_str(ops);
    s.push_str(&format!("    CK({}());\n", api.sync));
    for (i, b) in outputs.iter().enumerate() {
        s.push_str(&format!(
            "    CK({}(__hout{i}, {}, {}, {}));\n",
            api.memcpy,
            b.name,
            bytes(b),
            api.device_to_host
        ));
    }
    s.push_str("    unsigned char __ok = 0;\n");
    s.push_str("    if (fwrite(&__ok, 1, 1, stdout) != 1) break;\n");
    s.push_str("    fflush(stdout);\n");
    s.push_str("  }\n");
    s.push_str("  return 0;\n");
    s.push_str("}\n");
    s
}
