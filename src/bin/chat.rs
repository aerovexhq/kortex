//! Kortex Interactive Chat REPL binary.

use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_arg = args.get(1).map(|s| s.as_str());
    kortex::chat::run_chat_repl(model_arg)
}
