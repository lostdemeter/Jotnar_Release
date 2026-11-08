//! jotnar-press: Markdown -> print-ready HTML -> PDF.
//!
//! The press is deliberately small. It reads a book manifest, renders each
//! chapter with a handful of zine directives, generates the cover, masthead,
//! table of contents, part dividers, and the mnemonic register, then hands
//! the HTML to Paged.js and Chromium.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use pulldown_cmark::{html, Options, Parser};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Book {
    book: Meta,
    part: Vec<Part>,
}

#[derive(Debug, Deserialize)]
struct Meta {
    title: String,
    subtitle: String,
    issue: String,
    edition: String,
    date: String,
    tagline: String,
    imprint: String,
    author: String,
}

#[derive(Debug, Deserialize)]
struct Part {
    id: String,
    number: String,
    title: String,
    deck: String,
    #[serde(default)]
    theme: Option<String>,
    chapters: Vec<String>,
}

#[derive(Debug, Default)]
struct Front {
    title: String,
    number: String,
    kicker: String,
    lede: String,
    columns: u8,
    feature: bool,
    theme: String,
    art: String,
    art_caption: String,
    stamp: String,
}

struct Chapter {
    front: Front,
    body: String,
    id: String,
}

struct LoadedPart {
    part: Part,
    chapters: Vec<Chapter>,
}

#[derive(Clone)]
struct Op {
    name: String,
    arity_in: String,
    arity_out: String,
    sig: String,
    family: String,
    desc: String,
}

type R<T> = Result<T, String>;

fn main() {
    if let Err(e) = run() {
        eprintln!("jotnar-press: {e}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let mut book_path = PathBuf::from("book.toml");
    let mut out_pdf = PathBuf::from("out/jotnar-field-manual.pdf");
    let mut html_only = false;
    let mut no_pdf = false;
    let mut preview = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--book" => book_path = PathBuf::from(args.next().ok_or("--book needs a path")?),
            "--out" => out_pdf = PathBuf::from(args.next().ok_or("--out needs a path")?),
            "--html-only" => html_only = true,
            "--no-pdf" => no_pdf = true,
            "--preview" => preview = true,
            "-h" | "--help" => {
                println!(
                    "jotnar-press [--book book.toml] [--out out/manual.pdf] \
                     [--html-only] [--no-pdf] [--preview]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }

    let root = book_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let book_text = fs::read_to_string(&book_path)
        .map_err(|e| format!("cannot read {}: {e}", book_path.display()))?;
    let book: Book = toml::from_str(&book_text).map_err(|e| format!("book.toml: {e}"))?;
    let ops = load_ops(&root.join("data/ops.tsv"))?;

    let mut loaded = Vec::new();
    for part in &book.part {
        let mut chapters = Vec::new();
        for file in &part.chapters {
            let path = root.join(file);
            let text = fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            let (front, body) = split_frontmatter(&text);
            if front.title.is_empty() {
                return Err(format!("{}: missing title in frontmatter", path.display()));
            }
            let id = format!("ch-{}", slug(&front.number));
            chapters.push(Chapter {
                front,
                body: add_dropcap(&render_md(&body, &ops)?),
                id,
            });
        }
        loaded.push(LoadedPart {
            part: part.clone_for_render(),
            chapters,
        });
    }

    let css = fs::read_to_string(root.join("theme/book.css"))
        .map_err(|e| format!("cannot read theme/book.css: {e}"))?;
    let cover = cover_html(&book.book, &loaded, ops.len());
    let masthead = masthead_html(&book.book);
    let toc = toc_html(&loaded);
    let mut body = String::new();
    for lp in &loaded {
        body.push_str(&part_divider_html(&lp.part, &book.book));
        for ch in &lp.chapters {
            body.push_str(&chapter_html(ch, &lp.part, &book));
        }
    }

    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="author" content="{author}">
<meta name="description" content="{description}">
<style>
{css}
</style>
</head>
<body>
{cover}
{masthead}
{toc}
{body}
<script>
window.PagedConfig = {{
  auto: true,
  after: function () {{
    try {{
      var pages = Array.prototype.slice.call(document.querySelectorAll('.pagedjs_page'));
      var pageOf = new Map();
      pages.forEach(function (page, idx) {{
        page.querySelectorAll('[id]').forEach(function (el) {{
          if (!pageOf.has(el.id)) pageOf.set(el.id, idx + 1);
        }});
      }});
      document.querySelectorAll('.toc-page[data-target]').forEach(function (el) {{
        var n = pageOf.get(el.getAttribute('data-target'));
        if (n) el.textContent = String(n).padStart(2, '0');
      }});
      document.body.setAttribute('data-press-ready', '1');
    }} catch (e) {{
      console.error('jotnar-press toc stamp:', e);
    }}
  }}
}};
</script>
<script src="../press/vendor/paged.polyfill.js"></script>
</body>
</html>
"#,
        title = escape(&book.book.title),
        author = escape(&book.book.author),
        description = escape(&book.book.tagline),
        css = css
    );

    let out_dir = root.join("out");
    fs::create_dir_all(&out_dir).map_err(|e| format!("cannot create out/: {e}"))?;
    let html_path = out_dir.join("manual.html");
    // The HTML lives in out/, so asset URLs need one level up. We avoid a
    // <base> element: Chromium's file:// loader is unreliable with a base
    // that points above the document when the Paged.js script is relative.
    let html = html.replace("src=\"assets/", "src=\"../assets/");
    fs::write(&html_path, html).map_err(|e| format!("cannot write html: {e}"))?;
    println!("wrote {}", html_path.display());

    if html_only || no_pdf {
        return Ok(());
    }

    let abs_html = html_path
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize {}: {e}", html_path.display()))?;
    let abs_pdf = if out_pdf.is_absolute() {
        out_pdf.clone()
    } else {
        root.join(&out_pdf)
    };
    if let Some(parent) = abs_pdf.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let print_script = root.join("press/print_pdf.py");
    let status = Command::new("python3")
        .arg(&print_script)
        .arg(&abs_html)
        .arg(&abs_pdf)
        .status()
        .map_err(|e| format!("failed to run python3 {}: {e}", print_script.display()))?;
    if !status.success() {
        return Err(format!("print_pdf.py exited with {status}"));
    }
    println!("wrote {}", abs_pdf.display());
    if let Ok(meta) = fs::metadata(&abs_pdf) {
        println!("pdf: {} bytes", meta.len());
    }

    if preview {
        let preview_dir = out_dir.join("preview");
        let _ = fs::remove_dir_all(&preview_dir);
        let _ = fs::create_dir_all(&preview_dir);
        let _ = Command::new("pdftoppm")
            .args(["-png", "-r", "72"])
            .arg(&abs_pdf)
            .arg(preview_dir.join("page"))
            .status();
        println!("preview: {}", preview_dir.display());
    }
    Ok(())
}

impl Part {
    fn clone_for_render(&self) -> Part {
        Part {
            id: self.id.clone(),
            number: self.number.clone(),
            title: self.title.clone(),
            deck: self.deck.clone(),
            theme: self.theme.clone(),
            chapters: self.chapters.clone(),
        }
    }
}

fn load_ops(path: &Path) -> R<Vec<Op>> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 6 {
            return Err(format!("{}:{}: expected 6 fields", path.display(), n + 1));
        }
        out.push(Op {
            name: f[0].to_string(),
            arity_in: f[1].to_string(),
            arity_out: f[2].to_string(),
            sig: f[3].to_string(),
            family: f[4].to_string(),
            desc: f[5].to_string(),
        });
    }
    Ok(out)
}

fn split_frontmatter(text: &str) -> (Front, String) {
    let mut front = Front {
        columns: 2,
        theme: "paper".to_string(),
        ..Default::default()
    };
    if !text.starts_with("---\n") {
        return (front, text.to_string());
    }
    let rest = &text[4..];
    let Some(end) = rest.find("\n---\n") else {
        return (front, text.to_string());
    };
    let header = &rest[..end];
    let body = rest[end + 5..].to_string();
    for line in header.lines() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').to_string();
        match k {
            "title" => front.title = v,
            "number" => front.number = v,
            "kicker" => front.kicker = v,
            "lede" => front.lede = v,
            "columns" => front.columns = v.parse().unwrap_or(2),
            "feature" => front.feature = v == "true" || v == "yes",
            "theme" => front.theme = v,
            "art" => front.art = v,
            "art_caption" => front.art_caption = v,
            "stamp" => front.stamp = v,
            _ => {}
        }
    }
    (front, body)
}

fn add_dropcap(html: &str) -> String {
    if let Some(p) = html.find("<p>") {
        let start = p + 3;
        let rest = &html[start..];
        if let Some(first) = rest.chars().next() {
            if first.is_alphabetic() {
                let len = first.len_utf8();
                return format!(
                    "{}<p><span class=\"dropcap\">{}</span>{}",
                    &html[..p],
                    first,
                    &rest[len..]
                );
            }
        }
    }
    html.to_string()
}

fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Markdown + zine directives
// ---------------------------------------------------------------------------

fn render_md(src: &str, ops: &[Op]) -> R<String> {
    let mut out = String::new();
    let mut buf = String::new();
    let mut lines = src.lines().peekable();
    let mut fence = false;

    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            fence = !fence;
            buf.push_str(line);
            buf.push('\n');
            continue;
        }
        if !fence && trimmed.starts_with(":::") && !trimmed.starts_with("::::") {
            out.push_str(&md_to_html(&buf, ops));
            buf.clear();
            let header = trimmed.trim_start_matches(":::").trim();
            if header.is_empty() {
                continue;
            }
            let (name, attrs, title) = parse_directive(header);
            let mut body = String::new();
            let mut depth = 1usize;
            let mut inner_fence = false;
            let mut closed = false;
            while let Some(l) = lines.next() {
                let t = l.trim_start();
                if t.starts_with("```") {
                    inner_fence = !inner_fence;
                } else if !inner_fence && t.starts_with(":::") && !t.starts_with("::::") {
                    let rest = t.trim_start_matches(":::").trim();
                    if rest.is_empty() {
                        depth -= 1;
                        if depth == 0 {
                            closed = true;
                            break;
                        }
                    } else {
                        depth += 1;
                    }
                }
                body.push_str(l);
                body.push('\n');
            }
            if !closed {
                return Err(format!("unclosed ::: {name} block"));
            }
            out.push_str(&render_block(&name, &attrs, &title, &body, ops)?);
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    out.push_str(&md_to_html(&buf, ops));
    Ok(out)
}

fn md_to_html(src: &str, _ops: &[Op]) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_FOOTNOTES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);
    opts.insert(Options::ENABLE_SMART_PUNCTUATION);
    let parser = Parser::new_ext(src, opts);
    let mut html_out = String::new();
    html::push_html(&mut html_out, parser);
    highlight_code(&html_out)
}

fn parse_directive(line: &str) -> (String, HashMap<String, String>, String) {
    let mut rest = line.trim().to_string();
    let name_end = rest
        .find(|c: char| c.is_whitespace() || c == '[')
        .unwrap_or(rest.len());
    let name = rest[..name_end].to_ascii_lowercase();
    rest = rest[name_end..].to_string();
    let mut title = String::new();
    if let (Some(i), Some(j)) = (rest.find('['), rest.find(']')) {
        if j > i {
            title = rest[i + 1..j].trim().to_string();
            rest = format!("{}{}", &rest[..i], &rest[j + 1..]);
        }
    }
    let mut attrs = HashMap::new();
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        let start = i;
        while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '-')
        {
            i += 1;
        }
        let key: String = chars[start..i].iter().collect();
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i < chars.len() && chars[i] == '=' {
            i += 1;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            let quote = if i < chars.len() && (chars[i] == '"' || chars[i] == '\'') {
                let q = chars[i];
                i += 1;
                Some(q)
            } else {
                None
            };
            let vstart = i;
            if let Some(q) = quote {
                while i < chars.len() && chars[i] != q {
                    i += 1;
                }
            } else {
                while i < chars.len() && !chars[i].is_whitespace() {
                    i += 1;
                }
            }
            let val: String = chars[vstart..i].iter().collect();
            if i < chars.len() {
                i += 1;
            }
            if !key.is_empty() {
                attrs.insert(key, val);
            }
        } else if !key.is_empty() {
            attrs.insert(key, "true".to_string());
        } else {
            i += 1;
        }
    }
    (name, attrs, title)
}

fn render_block(
    name: &str,
    attrs: &HashMap<String, String>,
    title: &str,
    body: &str,
    ops: &[Op],
) -> R<String> {
    let inner = render_md(body, ops)?;
    let label = if !title.is_empty() {
        title.to_string()
    } else {
        attrs.get("title").cloned().unwrap_or_default()
    };
    let span = "span-all";
    Ok(match name {
        "note" => format!(
            r#"<aside class="note {span}"><div class="aside-title">{}</div>{inner}</aside>"#,
            escape(&label)
        ),
        "warning" => format!(
            r#"<aside class="warning {span}"><div class="aside-title">{}</div>{inner}</aside>"#,
            escape(&label)
        ),
        "sidebar" => format!(
            r#"<aside class="sidebar {span}"><div class="aside-title">{}</div>{inner}</aside>"#,
            escape(&label)
        ),
        "pullquote" => {
            let class = attrs.get("class").cloned().unwrap_or_default();
            format!(
                r#"<blockquote class="pullquote {class}">{inner}<cite>{}</cite></blockquote>"#,
                escape(&label)
            )
        }
        "spec" => format!(r#"<div class="spec {span}">{inner}</div>"#),
        "figure" => {
            let src = attrs.get("src").cloned().unwrap_or_default();
            let caption = attrs
                .get("caption")
                .cloned()
                .unwrap_or_else(|| label.clone());
            let class = attrs.get("class").cloned().unwrap_or_default();
            let img = if !src.is_empty() && inner.trim().is_empty() {
                format!(r#"<img src="{}" alt="">"#, escape(&src))
            } else {
                inner
            };
            format!(
                r#"<figure class="fig {class} {span}">{img}<figcaption>{}</figcaption></figure>"#,
                escape(&caption)
            )
        }
        "listing" => format!(
            r#"<div class="listing {span}"><div class="listing-label">{}</div>{inner}</div>"#,
            escape(&label)
        ),
        "op-table" => {
            let family = attrs
                .get("family")
                .cloned()
                .unwrap_or_else(|| "all".to_string());
            render_ops_table(ops, &family)
        }
        "end" => r#"<div class="endmark"><span>END</span></div>"#.to_string(),
        "pagebreak" => r#"<div class="pagebreak"></div>"#.to_string(),
        "stamp" => format!(
            r#"<div class="stamp">{}</div>"#,
            escape(if label.is_empty() { "DIGITAL" } else { &label })
        ),
        "raw" => inner,
        _ => format!(r#"<div class="block {name} {span}">{inner}</div>"#),
    })
}

fn render_ops_table(ops: &[Op], family: &str) -> String {
    let families = ["int", "linalg", "move", "image", "bytes"];
    let mut out = String::from(r#"<div class="op-register">"#);
    for fam in families {
        if family != "all" && family != fam {
            continue;
        }
        let rows: Vec<&Op> = ops.iter().filter(|o| o.family == fam).collect();
        if rows.is_empty() {
            continue;
        }
        out.push_str(&format!(
            r#"<h3 class="op-family">{}</h3><table class="ops"><thead><tr><th>Mnemonic</th><th>In→Out</th><th>Signature</th><th>What it does</th></tr></thead><tbody>"#,
            family_title(fam)
        ));
        for op in rows {
            out.push_str(&format!(
                r#"<tr><td class="op-name">{}</td><td class="op-arity">{}→{}</td><td class="op-sig">{}</td><td>{}</td></tr>"#,
                escape(&op.name),
                escape(&op.arity_in),
                escape(&op.arity_out),
                escape(&op.sig),
                escape(&op.desc)
            ));
        }
        out.push_str("</tbody></table>");
    }
    out.push_str("</div>");
    out
}

fn family_title(fam: &str) -> &'static str {
    match fam {
        "int" => "EXACT INTEGERS",
        "linalg" => "LINEAR ALGEBRA & THE TRANSFORMER",
        "move" => "MOVES, MASKS & SHAPES",
        "image" => "IMAGES & FIELDS",
        "bytes" => "BYTES & CONTAINERS",
        _ => "MNEMONICS",
    }
}

// ---------------------------------------------------------------------------
// Tiny syntax highlighter
// ---------------------------------------------------------------------------

fn highlight_code(html: &str) -> String {
    let mut out = String::new();
    let mut rest = html;
    const OPEN: &str = "<pre><code class=\"language-";
    const CLOSE: &str = "</code></pre>";
    while let Some(start) = rest.find(OPEN) {
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        let lang_start = OPEN.len();
        let Some(lang_end_rel) = after[lang_start..].find('"') else {
            out.push_str(after);
            return out;
        };
        let lang_end = lang_start + lang_end_rel;
        let lang = &after[lang_start..lang_end];
        let code_pos = after.find("<code").unwrap_or(0);
        let Some(gt_rel) = after[code_pos..].find('>') else {
            out.push_str(after);
            return out;
        };
        let gt = code_pos + gt_rel;
        let Some(close) = after.find(CLOSE) else {
            out.push_str(after);
            return out;
        };
        out.push_str(&after[..gt + 1]);
        out.push_str(&highlight_body(&after[gt + 1..close], lang));
        out.push_str(CLOSE);
        rest = &after[close + CLOSE.len()..];
    }
    out.push_str(rest);
    out
}

fn highlight_body(src: &str, lang: &str) -> String {
    match lang {
        "asm" => highlight_asm(src),
        "json" | "sh" | "bash" | "rust" | "c" | "toml" => highlight_generic(src, lang),
        _ => src.to_string(),
    }
}

fn highlight_asm(src: &str) -> String {
    let mut out = String::new();
    for line in src.split_inclusive('\n') {
        let (code, newline) = match line.strip_suffix('\n') {
            Some(c) => (c, "\n"),
            None => (line, ""),
        };
        let (before, comment) = match code.find('#') {
            Some(i) => (&code[..i], Some(&code[i..])),
            None => (code, None),
        };
        let mut i = 0usize;
        let chars: Vec<char> = before.chars().collect();
        while i < chars.len() {
            let c = chars[i];
            if c == '&' {
                let mut j = i;
                while j < chars.len() && chars[j] != ';' {
                    j += 1;
                }
                if j < chars.len() {
                    j += 1;
                }
                out.extend(&chars[i..j]);
                i = j;
            } else if c.is_ascii_uppercase() {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_uppercase()
                        || chars[i].is_ascii_digit()
                        || chars[i] == '_')
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                if i < chars.len() && chars[i] == '(' {
                    out.push_str("<span class=\"mn\">");
                    out.push_str(&word);
                    out.push_str("</span>");
                } else {
                    out.push_str(&word);
                }
            } else if c.is_ascii_digit() {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'x'
                        || chars[i].is_ascii_hexdigit())
                {
                    i += 1;
                }
                out.push_str("<span class=\"num\">");
                out.extend(&chars[start..i]);
                out.push_str("</span>");
            } else {
                out.push(c);
                i += 1;
            }
        }
        if let Some(c) = comment {
            out.push_str("<span class=\"cm\">");
            out.push_str(c);
            out.push_str("</span>");
        }
        out.push_str(newline);
    }
    out
}

fn highlight_generic(src: &str, lang: &str) -> String {
    let comment = if lang == "sh" || lang == "bash" || lang == "toml" {
        '#'
    } else {
        '/'
    };
    let mut out = String::new();
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' || c == '\'' {
            let q = c;
            let start = i;
            i += 1;
            while i < chars.len() && chars[i] != q {
                i += 1;
            }
            if i < chars.len() {
                i += 1;
            }
            out.push_str("<span class=\"str\">");
            out.extend(&chars[start..i]);
            out.push_str("</span>");
        } else if comment == '#' && c == '#' {
            let start = i;
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            out.push_str("<span class=\"cm\">");
            out.extend(&chars[start..i]);
            out.push_str("</span>");
        } else if comment == '/' && c == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            let start = i;
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            out.push_str("<span class=\"cm\">");
            out.extend(&chars[start..i]);
            out.push_str("</span>");
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Page furniture
// ---------------------------------------------------------------------------

fn cover_html(m: &Meta, loaded: &[LoadedPart], mnemonic_count: usize) -> String {
    let inside: String = loaded
        .iter()
        .filter(|p| p.part.number != "00")
        .map(|p| {
            format!(
                "<div class=\"cover-index-row\"><span>{}</span>{}</div>",
                escape(&p.part.number),
                escape(&p.part.title)
            )
        })
        .collect();
    format!(
        r#"<section class="cover" id="cover">
  <div class="cover-left">
    <div class="cover-issue">{issue} / {date}</div>
    <div class="cover-vertical">JOTNAR-LAN · JOTNAR-MAC · JOTNAR-WIT</div>
    <div class="cover-barcode"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
  </div>
  <div class="cover-mast">
    <div class="cover-plate">{title}</div>
    <div class="cover-title">FIELD<br>MANUAL</div>
    <div class="cover-sub">{subtitle}</div>
  </div>
  <div class="cover-art"><img src="assets/cover-art.svg" alt=""></div>
  <div class="cover-tagline">{tagline}</div>
  <div class="cover-box">
    <div class="cover-box-title">INSIDE THIS ISSUE</div>
    {inside}
    <div class="cover-index-row"><span>+</span>THE {mnemonic_count}-MNEMONIC REGISTER</div>
  </div>
  <div class="cover-stamp">DIGITAL<br><b>FIRST DRAFT</b></div>
  <div class="cover-foot">{imprint} / {author}</div>
</section>"#,
        issue = escape(&m.issue),
        date = escape(&m.date),
        title = escape(&m.title),
        subtitle = escape(&m.subtitle),
        tagline = escape(&m.tagline),
        imprint = escape(&m.imprint),
        author = escape(&m.author),
        inside = inside,
        mnemonic_count = mnemonic_count
    )
}

fn masthead_html(m: &Meta) -> String {
    format!(
        r#"<section class="masthead" id="masthead">
  <div class="masthead-rail">
    <div class="masthead-rail-text">BROUGHT TO YOU BY THE LATTICE</div>
  </div>
  <div class="masthead-main">
    <div class="masthead-top"><span>{issue} / {edition}</span><span>{date}</span></div>
    <div class="masthead-word">{title}</div>
    <div class="masthead-sub">{subtitle}</div>
    <div class="masthead-grid">
      <div>
        <h3>EDITOR</h3>
        <p>The Jotnar Working Group</p>
        <h3>ORACLE</h3>
        <p>The Python reference implementation</p>
      </div>
      <div>
        <h3>MACHINES</h3>
        <p>C99 · non-FPU · CUDA · Verilog · RV32IM</p>
        <h3>WITNESS</h3>
        <p>Differential parity, byte for byte</p>
      </div>
      <div>
        <h3>TYPE</h3>
        <p>Anton · Barlow Condensed · Courier Prime · IBM Plex Mono · Stardos Stencil</p>
        <h3>SETTING</h3>
        <p>Markdown, Paged.js, Chromium</p>
      </div>
    </div>
    <div class="masthead-warning">
      <b>FAIL-LOUD NOTICE.</b> Unknown mnemonics, arity mismatches, use-before-def,
      layout conflicts, and shape mismatches are errors, never silent guesses.
      If this manual is wrong, the assembler is the authority.
    </div>
    <div class="masthead-foot">
      <span>{imprint}</span>
      <span>GPL-3.0-ONLY CODE / CC BY-SA 4.0 TEXT</span>
      <span>NO REFERENCE TEXT OR LOGOS REPRODUCED</span>
    </div>
  </div>
</section>"#,
        issue = escape(&m.issue),
        edition = escape(&m.edition),
        date = escape(&m.date),
        title = escape(&m.title),
        subtitle = escape(&m.subtitle),
        imprint = escape(&m.imprint),
    )
}

fn toc_html(loaded: &[LoadedPart]) -> String {
    let mut parts = String::new();
    let mut n = 0usize;
    for lp in loaded {
        n += 1;
        parts.push_str(&format!(
            r#"<div class="toc-part">
  <div class="toc-part-head"><span>{}</span>{}</div>"#,
            escape(&lp.part.number),
            escape(&lp.part.title)
        ));
        for ch in &lp.chapters {
            parts.push_str(&format!(
                r##"<a class="toc-entry" href="#{}"><span class="toc-no">{}</span><span class="toc-title">{}</span><span class="toc-page" data-target="{}">00</span></a>"##,
                ch.id,
                escape(&ch.front.number),
                escape(&ch.front.title),
                ch.id
            ));
        }
        parts.push_str("</div>");
        if n == 3 {
            parts.push_str("<div class=\"toc-break\"></div>");
        }
    }
    format!(
        r##"<section class="toc" id="toc">
  <div class="toc-rail">
    <div class="toc-rail-top">JOTNAR<br>FIELD<br>MANUAL</div>
    <div class="toc-rail-mid">CONTENTS</div>
    <div class="toc-rail-bot">ISSUE 01<br>DIGITAL</div>
  </div>
  <div class="toc-main">
    <div class="toc-head">
      <div class="toc-kicker">EVERYTHING IN THIS ISSUE</div>
      <h1>CONTENTS</h1>
      <div class="toc-deck">The manual is a graph too. Follow the part labels, or jump straight to the register.</div>
    </div>
    <div class="toc-grid">{parts}</div>
  </div>
</section>"##
    )
}

fn part_divider_html(p: &Part, m: &Meta) -> String {
    let theme = p.theme.clone().unwrap_or_else(|| "dark".to_string());
    format!(
        r#"<section class="part-divider theme-{theme}" id="part-{id}">
  <div class="part-rail">{imprint}</div>
  <div class="part-number">{number}</div>
  <div class="part-title">{title}</div>
  <div class="part-deck">{deck}</div>
  <div class="part-rule"></div>
  <div class="part-stamp">{issue} / {date}</div>
</section>"#,
        theme = escape(&theme),
        id = escape(&p.id),
        imprint = escape(&m.imprint),
        number = escape(&p.number),
        title = escape(&p.title),
        deck = escape(&p.deck),
        issue = escape(&m.issue),
        date = escape(&m.date)
    )
}

fn chapter_html(ch: &Chapter, part: &Part, book: &Book) -> String {
    let f = &ch.front;
    let part_label = format!("PART {} / {}", part.number, part.title);
    let head = format!(
        r#"<header class="chapter-head">
  <div class="chapter-rule"><span class="chapter-kicker">{kicker}</span><span class="chapter-part">{part_label}</span></div>
  <div class="chapter-title-row">
    <span class="chapter-number">{number}</span>
    <h1 class="chapter-title">{title}</h1>
  </div>
  <p class="chapter-lede">{lede}</p>
</header>"#,
        kicker = escape(&f.kicker),
        part_label = escape(&part_label),
        number = escape(&f.number),
        title = escape(&f.title),
        lede = escape(&f.lede)
    );

    if f.feature {
        let theme = if f.theme.is_empty() {
            "red"
        } else {
            f.theme.as_str()
        };
        let art = if f.art.is_empty() {
            String::new()
        } else {
            format!(r#"<img class="opener-art" src="{}" alt="">"#, escape(&f.art))
        };
        let stamp = if f.stamp.is_empty() {
            format!("{} / {}", book.book.issue, book.book.edition)
        } else {
            f.stamp.clone()
        };
        return format!(
            r#"<section class="feature-opener theme-{theme}" id="{id}">
  <div class="opener-rail">{part_label}</div>
  <div class="opener-number">{number}</div>
  <h1 class="opener-title">{title}</h1>
  <p class="opener-lede">{lede}</p>
  {art}
  <div class="opener-stamp">{stamp}</div>
  <div class="opener-foot">{imprint} / {issue}</div>
</section>
<section class="chapter chapter-flow" id="{id}-body">
  <div class="flow-head"><span>{part_label}</span><b>{title}</b></div>
  <div class="chapter-body cols-{columns}">{body}</div>
</section>"#,
            theme = escape(theme),
            id = ch.id,
            part_label = escape(&part_label),
            number = escape(&f.number),
            title = escape(&f.title),
            lede = escape(&f.lede),
            art = art,
            stamp = escape(&stamp),
            imprint = escape(&book.book.imprint),
            issue = escape(&book.book.issue),
            columns = f.columns,
            body = ch.body
        );
    }

    format!(
        r#"<section class="chapter" id="{id}">
{head}
<div class="chapter-body cols-{columns}">{body}</div>
</section>"#,
        id = ch.id,
        head = head,
        columns = f.columns,
        body = ch.body
    )
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
