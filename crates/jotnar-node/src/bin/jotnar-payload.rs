//! Command-line client for JotnarOS payload/event commands.
//!
//! Core programs/exec still use the upstream `jotnar-ctl`; this tool covers the
//! M3 supervisor surface:
//!
//! ```sh
//! jotnar-payload --addr HOST:9999 --token SECRET put heartbeat heartbeat.asm \
//!     --tick-ms 1000 --autostart --console both --seed seed.json
//! jotnar-payload ... start heartbeat
//! jotnar-payload ... status
//! jotnar-payload ... poll --after 0
//! jotnar-payload ... stop heartbeat
//! ```

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;

use jotnar::protocol::{decode_frame, request, PROTO_V};
use serde_json::{json, Value as Json};

struct Conn {
    stream: TcpStream,
    token: Option<String>,
    next_id: u64,
}

impl Conn {
    fn connect(addr: &str, token: Option<String>) -> Result<Self, String> {
        let stream =
            TcpStream::connect(addr).map_err(|e| format!("cannot connect to {addr}: {e}"))?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(60)));
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(60)));
        Ok(Self {
            stream,
            token,
            next_id: 1,
        })
    }

    fn roundtrip(
        &mut self,
        typ: &str,
        mut fields: Json,
        payload: &[u8],
    ) -> Result<(Json, Vec<u8>), String> {
        if let Some(t) = &self.token {
            if let Some(o) = fields.as_object_mut() {
                o.insert("auth".into(), json!(t));
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        let frame = request(id, typ, fields, payload).map_err(|e| e.to_string())?;
        self.stream
            .write_all(&frame)
            .map_err(|e| format!("write: {e}"))?;
        let mut hdr = [0u8; 4];
        self.stream
            .read_exact(&mut hdr)
            .map_err(|e| format!("read header: {e}"))?;
        let n = u32::from_be_bytes(hdr) as usize;
        if n > 256 * 1024 * 1024 {
            return Err(format!("response too large: {n}"));
        }
        let mut body = vec![0u8; n];
        self.stream
            .read_exact(&mut body)
            .map_err(|e| format!("read body: {e}"))?;
        let mut full = Vec::with_capacity(4 + n);
        full.extend_from_slice(&hdr);
        full.extend_from_slice(&body);
        let (h, p) = decode_frame(&full).map_err(|e| e.to_string())?;
        if h.get("id").and_then(|x| x.as_u64()) != Some(id) {
            return Err(format!("response id mismatch: {h}"));
        }
        if h.get("v").and_then(|x| x.as_u64()) != Some(PROTO_V) {
            return Err(format!("response version mismatch: {h}"));
        }
        Ok((h, p.to_vec()))
    }

    fn call(&mut self, typ: &str, fields: Json, payload: &[u8]) -> Result<Json, String> {
        let (h, p) = self.roundtrip(typ, fields, payload)?;
        if h.get("ok").and_then(|x| x.as_bool()) != Some(true) {
            return Err(format!(
                "node error: {}",
                h.get("error").cloned().unwrap_or(json!("?"))
            ));
        }
        if typ == "event-poll" {
            return Ok(attach_event_text(h, &p));
        }
        Ok(h)
    }
}

fn attach_event_text(mut h: Json, payload: &[u8]) -> Json {
    if let Some(events) = h.get_mut("events").and_then(|x| x.as_array_mut()) {
        for ev in events {
            let off = ev.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let len = ev.get("len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            if off
                .checked_add(len)
                .map(|e| e <= payload.len())
                .unwrap_or(false)
            {
                let bytes = &payload[off..off + len];
                let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                if let Some(o) = ev.as_object_mut() {
                    o.insert("hex".into(), json!(hex));
                }
                if let Ok(s) = std::str::from_utf8(bytes) {
                    if let Some(o) = ev.as_object_mut() {
                        o.insert("text".into(), json!(s));
                    }
                }
            }
        }
    }
    h
}

fn usage() -> &'static str {
    "jotnar-payload - control JotnarOS payloads\n\
     \n\
     usage:\n\
       jotnar-payload [--addr ADDR] [--token SECRET|--token-file FILE] ping\n\
       jotnar-payload ... put <name> <file.asm> [--tick-ms N] [--autostart]\n\
             [--console both|local|serial|none] [--seed payload.json] [--files a,b]\n\
       jotnar-payload ... start|stop|reload|drop <name>\n\
       jotnar-payload ... status [name]\n\
       jotnar-payload ... poll [--after N] [--max N] [--kinds a,b]\n\
       jotnar-payload ... write <name> --data TEXT | --file FILE\n\
             [--target stdin|file] [--path REL] [--mode write|append]\n\
       jotnar-payload ... read <name> --path REL\n\
       jotnar-payload ... bundle-put <name> <rootfs.tar.gz>\n\
       jotnar-payload ... console-start <name> [--vt N]\n\
       jotnar-payload ... console-stop <name>\n\
       jotnar-payload ... console-write <name> --data TEXT | --file FILE\n\
       jotnar-payload ... console-restart <name> [--vt N]\n\
       jotnar-payload ... console-status [name] | console-list\n\
       jotnar-payload ... module-event <name> <event>\n\
       jotnar-payload ... module-step <name> <event>   (interpreted oracle)\n\
       jotnar-payload ... module-step-emitted <name> <event>\n\
       jotnar-payload ... module-event-interpreted <name> <event>\n
       jotnar-payload ... module-status [name] | module-list\n\
     \n\
     --addr defaults to 127.0.0.1:9999.\n"
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut addr = "127.0.0.1:9999".to_string();
    let mut token: Option<String> = None;
    let mut token_file: Option<PathBuf> = None;
    while let Some(a) = args.first().cloned() {
        match a.as_str() {
            "--addr" => {
                args.remove(0);
                addr = args.first().cloned().ok_or("--addr needs a value")?;
                args.remove(0);
            }
            "--token" => {
                args.remove(0);
                token = Some(args.first().cloned().ok_or("--token needs a value")?);
                args.remove(0);
            }
            "--token-file" => {
                args.remove(0);
                token_file = Some(PathBuf::from(
                    args.first().cloned().ok_or("--token-file needs a value")?,
                ));
                args.remove(0);
            }
            _ => break,
        }
    }
    if token.is_none() {
        if let Some(p) = token_file {
            token = Some(
                std::fs::read_to_string(&p)
                    .map_err(|e| format!("token file: {e}"))?
                    .trim()
                    .to_string(),
            );
        }
    }
    if args.is_empty() {
        return Err(usage().to_string());
    }
    let cmd = args.remove(0);
    let mut c = Conn::connect(&addr, token)?;

    match cmd.as_str() {
        "ping" => print_json(&c.call("ping", json!({}), &[])?),
        "put" => {
            if args.len() < 2 {
                return Err(format!("put needs <name> <file.asm>\n\n{}", usage()));
            }
            let name = args.remove(0);
            let file = args.remove(0);
            let mut tick_ms = 0u64;
            let mut autostart = false;
            let mut console = "both".to_string();
            let mut seed: Option<Json> = None;
            let mut files: Vec<String> = Vec::new();
            while let Some(a) = args.first().cloned() {
                args.remove(0);
                match a.as_str() {
                    "--tick-ms" => {
                        tick_ms = args
                            .first()
                            .cloned()
                            .ok_or("--tick-ms needs a value")?
                            .parse()
                            .map_err(|_| "bad --tick-ms")?;
                        args.remove(0);
                    }
                    "--autostart" => autostart = true,
                    "--console" => {
                        console = args.first().cloned().ok_or("--console needs a value")?;
                        args.remove(0);
                    }
                    "--files" => {
                        files = args
                            .first()
                            .cloned()
                            .ok_or("--files needs a list")?
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                        args.remove(0);
                    }
                    "--seed" => {
                        let f = args.first().cloned().ok_or("--seed needs a file")?;
                        args.remove(0);
                        let text = std::fs::read_to_string(&f).map_err(|e| format!("seed: {e}"))?;
                        seed = Some(
                            serde_json::from_str(&text).map_err(|e| format!("seed JSON: {e}"))?,
                        );
                    }
                    other => return Err(format!("unknown put argument {other:?}")),
                }
            }
            let listing = std::fs::read(&file).map_err(|e| format!("cannot read {file}: {e}"))?;
            let mut fields = json!({
                "name": name, "autostart": autostart, "tick_ms": tick_ms,
                "console": console, "restart": "never", "files": files
            });
            if let Some(s) = seed {
                fields.as_object_mut().unwrap().insert("seed".into(), s);
            }
            print_json(&c.call("payload-put", fields, &listing)?);
        }
        "start" | "stop" | "reload" | "drop" => {
            let name = args.first().cloned().ok_or("missing payload name")?;
            let typ = format!("payload-{cmd}");
            print_json(&c.call(&typ, json!({"name": name}), &[])?);
        }
        "status" => {
            let mut fields = json!({});
            if let Some(n) = args.first() {
                fields
                    .as_object_mut()
                    .unwrap()
                    .insert("name".into(), json!(n));
            }
            print_json(&c.call("payload-status", fields, &[])?);
        }
        "poll" => {
            let mut after = 0u64;
            let mut max = 64u64;
            let mut kinds: Vec<String> = Vec::new();
            while let Some(a) = args.first().cloned() {
                args.remove(0);
                match a.as_str() {
                    "--after" => {
                        after = args
                            .first()
                            .cloned()
                            .ok_or("--after needs a value")?
                            .parse()
                            .map_err(|_| "bad --after")?;
                        args.remove(0);
                    }
                    "--max" => {
                        max = args
                            .first()
                            .cloned()
                            .ok_or("--max needs a value")?
                            .parse()
                            .map_err(|_| "bad --max")?;
                        args.remove(0);
                    }
                    "--kinds" => {
                        kinds = args
                            .first()
                            .cloned()
                            .ok_or("--kinds needs a list")?
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                        args.remove(0);
                    }
                    other => return Err(format!("unknown poll argument {other:?}")),
                }
            }
            print_json(&c.call(
                "event-poll",
                json!({"after_seq": after, "max": max, "kinds": kinds}),
                &[],
            )?);
        }
        "write" => {
            let name = args.first().cloned().ok_or("write needs <name>")?;
            args.remove(0);
            let mut data = Vec::new();
            let mut target = "stdin".to_string();
            let mut path = String::new();
            let mut mode = "write".to_string();
            while let Some(a) = args.first().cloned() {
                args.remove(0);
                match a.as_str() {
                    "--data" => {
                        data = args
                            .first()
                            .cloned()
                            .ok_or("--data needs text")?
                            .into_bytes();
                        args.remove(0);
                    }
                    "--file" => {
                        let f = args.first().cloned().ok_or("--file needs a path")?;
                        args.remove(0);
                        data = std::fs::read(&f).map_err(|e| format!("read {f}: {e}"))?;
                    }
                    "--target" => {
                        target = args.first().cloned().ok_or("--target needs a value")?;
                        args.remove(0);
                    }
                    "--path" => {
                        path = args.first().cloned().ok_or("--path needs a value")?;
                        args.remove(0);
                    }
                    "--mode" => {
                        mode = args.first().cloned().ok_or("--mode needs a value")?;
                        args.remove(0);
                    }
                    other => return Err(format!("unknown write argument {other:?}")),
                }
            }
            let fields = json!({"name": name, "target": target, "path": path, "mode": mode});
            print_json(&c.call("event-write", fields, &data)?);
        }
        "read" => {
            let name = args.first().cloned().ok_or("read needs <name>")?;
            args.remove(0);
            let mut path = String::new();
            while let Some(a) = args.first().cloned() {
                args.remove(0);
                match a.as_str() {
                    "--path" => {
                        path = args.first().cloned().ok_or("--path needs a value")?;
                        args.remove(0);
                    }
                    other => return Err(format!("unknown read argument {other:?}")),
                }
            }
            let (mut h, payload) =
                c.roundtrip("file-read", json!({"name": name, "path": path}), &[])?;
            if h.get("ok").and_then(|x| x.as_bool()) != Some(true) {
                return Err(format!(
                    "node error: {}",
                    h.get("error").cloned().unwrap_or(json!("?"))
                ));
            }
            let hex: String = payload.iter().map(|b| format!("{b:02x}")).collect();
            if let Some(o) = h.as_object_mut() {
                o.insert("hex".into(), json!(hex));
                if let Ok(s) = std::str::from_utf8(&payload) {
                    o.insert("text".into(), json!(s));
                }
            }
            print_json(&h);
        }
        "bundle-put" => {
            if args.len() < 2 {
                return Err("bundle-put needs <name> <rootfs.tar.gz>".to_string());
            }
            let name = args.remove(0);
            let file = args.remove(0);
            let bytes = std::fs::read(&file).map_err(|e| format!("read {file}: {e}"))?;
            print_json(&c.call(
                "bundle-put",
                json!({"name": name, "kind": "console"}),
                &bytes,
            )?);
        }
        "console-start" => {
            let name = args.first().cloned().ok_or("console-start needs <name>")?;
            args.remove(0);
            let mut vt = 2u64;
            while let Some(a) = args.first().cloned() {
                args.remove(0);
                match a.as_str() {
                    "--vt" => {
                        vt = args
                            .first()
                            .cloned()
                            .ok_or("--vt needs a value")?
                            .parse()
                            .map_err(|_| "bad --vt")?;
                        args.remove(0);
                    }
                    other => return Err(format!("unknown console-start argument {other:?}")),
                }
            }
            print_json(&c.call("console-start", json!({"name": name, "vt": vt}), &[])?);
        }
        "console-stop" => {
            let name = args.first().cloned().ok_or("console-stop needs <name>")?;
            print_json(&c.call("console-stop", json!({"name": name}), &[])?);
        }
        "console-write" => {
            let name = args.first().cloned().ok_or("console-write needs <name>")?;
            args.remove(0);
            let mut data = Vec::new();
            while let Some(a) = args.first().cloned() {
                args.remove(0);
                match a.as_str() {
                    "--data" => {
                        data = args
                            .first()
                            .cloned()
                            .ok_or("--data needs text")?
                            .into_bytes();
                        args.remove(0);
                    }
                    "--file" => {
                        let f = args.first().cloned().ok_or("--file needs a path")?;
                        args.remove(0);
                        data = std::fs::read(&f).map_err(|e| format!("read {f}: {e}"))?;
                    }
                    other => return Err(format!("unknown console-write argument {other:?}")),
                }
            }
            print_json(&c.call("console-write", json!({"name": name}), &data)?);
        }
        "console-restart" => {
            let name = args
                .first()
                .cloned()
                .ok_or("console-restart needs <name>")?;
            args.remove(0);
            let mut vt = 2u64;
            while let Some(a) = args.first().cloned() {
                args.remove(0);
                match a.as_str() {
                    "--vt" => {
                        vt = args
                            .first()
                            .cloned()
                            .ok_or("--vt needs a value")?
                            .parse()
                            .map_err(|_| "bad --vt")?;
                        args.remove(0);
                    }
                    other => return Err(format!("unknown console-restart argument {other:?}")),
                }
            }
            print_json(&c.call("console-restart", json!({"name": name, "vt": vt}), &[])?);
        }
        "console-status" => {
            let mut fields = json!({});
            if let Some(n) = args.first() {
                fields
                    .as_object_mut()
                    .unwrap()
                    .insert("name".into(), json!(n));
            }
            print_json(&c.call("console-status", fields, &[])?);
        }
        "console-list" => {
            print_json(&c.call("console-list", json!({}), &[])?);
        }
        "module-event" | "module-step" | "module-step-emitted" | "module-event-interpreted" => {
            if args.len() < 2 {
                return Err(format!("{cmd} needs <name> <event>"));
            }
            let name = args.remove(0);
            let event = args.remove(0);
            let mut fields = json!({"module": name, "event": event});
            while let Some(a) = args.first().cloned() {
                match a.as_str() {
                    "--args" => {
                        args.remove(0);
                        let raw = args.first().cloned().ok_or("--args needs JSON")?;
                        args.remove(0);
                        let parsed: Json = serde_json::from_str(&raw)
                            .map_err(|e| format!("bad --args JSON: {e}"))?;
                        fields
                            .as_object_mut()
                            .unwrap()
                            .insert("args".into(), parsed);
                    }
                    other => return Err(format!("unknown argument {other:?}")),
                }
            }
            print_json(&c.call(&cmd, fields, &[])?);
        }
        "module-status" | "module-list" => {
            let mut fields = json!({});
            if let Some(n) = args.first() {
                fields
                    .as_object_mut()
                    .unwrap()
                    .insert("name".into(), json!(n));
            }
            print_json(&c.call("module-status", fields, &[])?);
        }
        "-h" | "--help" | "help" => return Err(usage().to_string()),
        other => return Err(format!("unknown command {other:?}\n\n{}", usage())),
    }
    Ok(())
}

fn print_json(v: &Json) {
    println!("{}", serde_json::to_string_pretty(v).unwrap());
}
