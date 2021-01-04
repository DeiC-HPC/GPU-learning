use crate::schema::*;
use anyhow::Result;
use structopt::StructOpt;

#[derive(Copy, Clone, Debug, StructOpt)]
pub enum Format {
    CppOpenAcc,
    CppOpenMp,
    Cuda,
    FortranOpenAcc,
    FortranOpenMp,
    Python,
}

impl Format {
    pub fn possible_values() -> &'static [&'static str] {
        &[
            "cpp_openacc",
            "cpp_openmp",
            "fortran_openacc",
            "fortran_openmp",
            "cuda",
            "python",
        ]
    }
    pub fn parse(s: &str) -> Result<Format> {
        match s {
            "cpp_openacc" => Ok(Format::CppOpenAcc),
            "cpp_openmp" => Ok(Format::CppOpenMp),
            "fortran_openacc" => Ok(Format::FortranOpenAcc),
            "fortran_openmp" => Ok(Format::FortranOpenMp),
            "cuda" => Ok(Format::Cuda),
            "python" => Ok(Format::Python),
            _ => anyhow::bail!("Unknown format {:?}", s),
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Format::CppOpenAcc => "C++ with OpenACC",
            Format::CppOpenMp => "C++ with OpenMP",
            Format::Cuda => "Cuda compiler",
            Format::FortranOpenAcc => "Fortran with OpenACC",
            Format::FortranOpenMp => "Fortran with OpenMP",
            Format::Python => "Python 3",
        }
    }

    fn language(self) -> &'static str {
        match self {
            Format::CppOpenAcc | Format::CppOpenMp | Format::Cuda => "c++",
            Format::FortranOpenAcc | Format::FortranOpenMp => "fortran",
            Format::Python => "python",
        }
    }

    fn kernel_name(self) -> &'static str {
        match self {
            Format::CppOpenAcc => "kernel_cpp_openacc",
            Format::CppOpenMp => "kernel_cpp_openmp",
            Format::Cuda => "kernel_cuda",
            Format::FortranOpenAcc => "kernel_fortran_openacc",
            Format::FortranOpenMp => "kernel_fortran_openmp",
            Format::Python => "python3",
        }
    }

    fn file_extension(self) -> &'static str {
        match self {
            Format::CppOpenAcc => ".cpp",
            Format::CppOpenMp => ".cpp",
            Format::Cuda => ".cu",
            Format::FortranOpenAcc => ".f90",
            Format::FortranOpenMp => ".f90",
            Format::Python => ".py",
        }
    }

    fn mimetype(self) -> &'static str {
        match self {
            Format::CppOpenAcc
            | Format::CppOpenMp
            | Format::Cuda
            | Format::FortranOpenAcc
            | Format::FortranOpenMp => "text/plain",
            Format::Python => "text/x-python",
        }
    }

    fn nbconvert_exporter(self) -> Option<String> {
        match self {
            Format::CppOpenAcc
            | Format::CppOpenMp
            | Format::Cuda
            | Format::FortranOpenAcc
            | Format::FortranOpenMp => None,
            Format::Python => Some("python".to_string()),
        }
    }

    fn pygments_lexer(self) -> Option<String> {
        match self {
            Format::CppOpenAcc
            | Format::CppOpenMp
            | Format::Cuda
            | Format::FortranOpenAcc
            | Format::FortranOpenMp => None,
            Format::Python => Some("ipython3".to_string()),
        }
    }

    fn codemirror_mode(self) -> Option<CodemirrorMode> {
        match self {
            Format::CppOpenAcc
            | Format::CppOpenMp
            | Format::Cuda
            | Format::FortranOpenAcc
            | Format::FortranOpenMp => None,
            Format::Python => Some(CodemirrorMode {
                name: "ipython".to_string(),
                version: 3,
            }),
        }
    }

    pub fn to_metadata(self) -> Metadata {
        Metadata {
            kernelspec: KernelSpec {
                display_name: self.display_name().to_string(),
                language: self.language().to_string(),
                name: self.kernel_name().to_string(),
            },
            language_info: LanguageInfo {
                file_extension: self.file_extension().to_string(),
                mimetype: self.mimetype().to_string(),
                name: self.language().to_string(),
                nbconvert_exporter: self.nbconvert_exporter(),
                pygments_lexer: self.pygments_lexer(),
                codemirror_mode: self.codemirror_mode(),
            },
        }
    }
}
