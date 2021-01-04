use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize, Debug)]
pub struct Notebook {
    pub cells: Vec<Cell>,
    pub metadata: Metadata,
    pub nbformat: u32,
    pub nbformat_minor: u32,
}

#[derive(Serialize, Debug, PartialEq, Eq, Hash)]
pub struct Unknown;

#[derive(Serialize, Debug)]
pub struct Cell {
    pub cell_type: String,
    pub execution_count: Option<Unknown>,
    pub metadata: HashMap<Unknown, Unknown>,
    pub outputs: Vec<Unknown>,
    pub source: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct Metadata {
    pub kernelspec: KernelSpec,
    pub language_info: LanguageInfo,
}

#[derive(Serialize, Debug)]
pub struct KernelSpec {
    pub display_name: String,
    pub language: String,
    pub name: String,
}

#[derive(Serialize, Debug)]
pub struct LanguageInfo {
    pub file_extension: String,
    pub mimetype: String,
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub codemirror_mode: Option<CodemirrorMode>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nbconvert_exporter: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pygments_lexer: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct CodemirrorMode {
    pub name: String,
    pub version: u32,
}
