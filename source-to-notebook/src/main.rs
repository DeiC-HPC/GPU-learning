mod known_formats;
mod schema;

use anyhow::{Context, Result};
use known_formats::Format;
use std::path::PathBuf;
use structopt::{clap::AppSettings, StructOpt};

#[derive(Debug, StructOpt)]
#[structopt(
    global_setting = AppSettings::ColoredHelp,
    global_setting = AppSettings::VersionlessSubcommands,
    name = "source-to-notebook",
    about = "Converts source fiels to jupyter notebooks",
)]
pub struct Args {
    #[structopt(parse(try_from_str = Format::parse), possible_values = &Format::possible_values())]
    /// The language used in the file
    pub format: Format,

    #[structopt(parse(from_os_str))]
    /// File to convert to a notebook
    pub input: PathBuf,

    #[structopt(parse(from_os_str))]
    /// Output path for the generated notebook
    pub output: PathBuf,
}

#[paw::main]
fn main(args: Args) -> Result<()> {
    let code = std::fs::read_to_string(args.input).context("Could not read file")?;
    let lines = code
        .trim()
        .lines()
        .map(|line| format!("{}\n", line))
        .collect::<Vec<String>>();
    let cells = vec![schema::Cell {
        cell_type: "code".to_string(),
        execution_count: None,
        metadata: Default::default(),
        outputs: vec![],
        source: lines,
    }];
    let notebook = schema::Notebook {
        cells,
        metadata: args.format.to_metadata(),
        nbformat: 4,
        nbformat_minor: 4,
    };

    serde_json::to_writer(
        std::fs::File::create(args.output).context("Could not create output file")?,
        &notebook,
    )
    .context("Could not write output file")?;

    Ok(())
}
