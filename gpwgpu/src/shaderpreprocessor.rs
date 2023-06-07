use std::{borrow::Cow, collections::HashMap, path::Path, fs};
use thiserror::Error;
use regex::Regex;
use once_cell::sync::Lazy;

#[derive(Clone, PartialEq, Debug)]
// #[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum ShaderDefVal {
    Bool(String, bool),
    Int(String, i32),
    UInt(String, u32),
    Any(String, String),
    Float(String, f32),
}

impl std::hash::Hash for ShaderDefVal{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self{
            ShaderDefVal::Bool(name, val) => {name.hash(state); val.hash(state)},
            ShaderDefVal::Int(name, val) => {name.hash(state); val.hash(state)},
            ShaderDefVal::UInt(name, val) => {name.hash(state); val.hash(state)},
            ShaderDefVal::Float(name, val) => {name.hash(state); val.to_bits().hash(state)},
            ShaderDefVal::Any(name, val) => {name.hash(state); val.hash(state)},
        }
    }
}


impl From<&str> for ShaderDefVal {
    fn from(key: &str) -> Self {
        ShaderDefVal::Bool(key.to_string(), true)
    }
}

impl From<String> for ShaderDefVal {
    fn from(key: String) -> Self {
        ShaderDefVal::Bool(key, true)
    }
}

impl ShaderDefVal {
    pub fn value_as_string(&self) -> String {
        match self {
            ShaderDefVal::Bool(_, def) => def.to_string(),
            ShaderDefVal::Int(_, def) => def.to_string(),
            ShaderDefVal::UInt(_, def) => def.to_string(),
            ShaderDefVal::Float(_, def) => def.to_string(),
            ShaderDefVal::Any(_, def) => def.to_string(),
        }
    }
}


#[derive(Debug, Clone)]
// #[uuid = "d95bc916-6c55-4de3-9622-37e7b6969fda"]
pub struct Shader {
    source: Source,
    import_path: Option<ShaderImport>,
    imports: Vec<ShaderImport>,
}

impl Shader {
    pub fn from_wgsl(source: impl Into<Cow<'static, str>>) -> Shader {
        let source = source.into();
        let shader_imports = SHADER_IMPORT_PROCESSOR.get_imports_from_str(&source);
        Shader {
            imports: shader_imports.imports,
            import_path: shader_imports.import_path,
            source: Source(source),
        }
    }

    pub fn set_import_path<P: Into<String>>(&mut self, import_path: P) {
        self.import_path = Some(ShaderImport::Custom(import_path.into()));
    }

    #[must_use]
    pub fn with_import_path<P: Into<String>>(mut self, import_path: P) -> Self {
        self.set_import_path(import_path);
        self
    }

    #[inline]
    pub fn import_path(&self) -> Option<&ShaderImport> {
        self.import_path.as_ref()
    }

    pub fn imports(&self) -> impl ExactSizeIterator<Item = &ShaderImport> {
        self.imports.iter()
    }
}

#[derive(Debug, Clone)]
pub struct Source(Cow<'static, str>);

/// A processed [Shader]. This cannot contain preprocessor directions. It must be "ready to compile"
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ProcessedShader(pub Cow<'static, str>);

impl ProcessedShader {
    pub fn get_source(&self) -> &str {
		&self.0
    }
    
}

#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum ProcessShaderError {
    #[error("Too many '# endif' lines. Each endif should be preceded by an if statement.")]
    TooManyEndIfs,
    #[error(
        "Not enough '# endif' lines. Each if statement should be followed by an endif statement."
    )]
    NotEnoughEndIfs,
    #[error("This Shader's format does not support processing shader defs.")]
    ShaderFormatDoesNotSupportShaderDefs,
    #[error("This Shader's format does not support imports.")]
    ShaderFormatDoesNotSupportImports,
    #[error("Unresolved import: {0:?}.")]
    UnresolvedImport(ShaderImport),
    #[error("The shader import {0:?} does not match the source file type. Support for this might be added in the future.")]
    MismatchedImportFormat(ShaderImport),
    #[error("Unknown shader def operator: '{operator}'")]
    UnknownShaderDefOperator { operator: String },
    #[error("Unknown shader def: '{shader_def_name}'")]
    UnknownShaderDef { shader_def_name: String },
    #[error(
        "Invalid shader def comparison for '{shader_def_name}': expected {expected}, got {value}"
    )]
    InvalidShaderDefComparisonValue {
        shader_def_name: String,
        expected: String,
        value: String,
    },
    #[error(
        "Invalid shader def comparison for '{shader_def_name}' with value '{value}': Only != and == are allowed for string comparisons"
    )]
    InvalidShaderDefComparisonAny {
        shader_def_name: String,
        value: String,
    },
    #[error("Invalid shader def definition for '{shader_def_name}': {value}")]
    InvalidShaderDefDefinitionValue {
        shader_def_name: String,
        value: String,
    },
}

pub struct ShaderImportProcessor {
    import_asset_path_regex: Regex,
    import_custom_path_regex: Regex,
    define_import_path_regex: Regex,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ShaderImport {
    AssetPath(String),
    Custom(String),
}

impl From<&str> for ShaderImport{
    fn from(value: &str) -> Self {
        Self::Custom(value.to_string())
    }
}

impl Default for ShaderImportProcessor {
    fn default() -> Self {
        Self {
            import_asset_path_regex: Regex::new(r#"^\s*#\s*import\s+"(.+)""#).unwrap(),
            import_custom_path_regex: Regex::new(r"^\s*#\s*import\s+(.+)").unwrap(),
            define_import_path_regex: Regex::new(r"^\s*#\s*define_import_path\s+(.+)").unwrap(),
        }
    }
}

#[derive(Default)]
pub struct ShaderImports {
    imports: Vec<ShaderImport>,
    import_path: Option<ShaderImport>,
}

impl ShaderImportProcessor {
    pub fn get_imports(&self, shader: &Shader) -> ShaderImports {
		self.get_imports_from_str(&shader.source.0)
        // match &shader.source {
        //     Source::Wgsl(source) => self.get_imports_from_str(source),
        //     Source::Glsl(source, _stage) => self.get_imports_from_str(source),
        //     Source::SpirV(_source) => ShaderImports::default(),
        // }
    }

    pub fn get_imports_from_str(&self, shader: &str) -> ShaderImports {
        let mut shader_imports = ShaderImports::default();
        for line in shader.lines() {
            if let Some(cap) = self.import_asset_path_regex.captures(line) {
                let import = cap.get(1).unwrap();
                shader_imports
                    .imports
                    .push(ShaderImport::AssetPath(import.as_str().to_string()));
            } else if let Some(cap) = self.import_custom_path_regex.captures(line) {
                let import = cap.get(1).unwrap();
                shader_imports
                    .imports
                    .push(ShaderImport::Custom(import.as_str().to_string()));
            } else if let Some(cap) = self.define_import_path_regex.captures(line) {
                let path = cap.get(1).unwrap();
                shader_imports.import_path = Some(ShaderImport::Custom(path.as_str().to_string()));
            }
        }

        shader_imports
    }
}

pub static SHADER_IMPORT_PROCESSOR: Lazy<ShaderImportProcessor> =
    Lazy::new(ShaderImportProcessor::default);

pub struct ShaderProcessor {
    ifdef_regex: Regex,
    ifndef_regex: Regex,
    ifop_regex: Regex,
    else_ifdef_regex: Regex,
    else_regex: Regex,
    endif_regex: Regex,
    define_regex: Regex,
    def_regex: Regex,
    def_regex_delimited: Regex,
}
impl Default for ShaderProcessor {
    fn default() -> Self {
        Self {
            ifdef_regex: Regex::new(r"^\s*#\s*ifdef\s*([\w|\d|_]+)").unwrap(),
            ifndef_regex: Regex::new(r"^\s*#\s*ifndef\s*([\w|\d|_]+)").unwrap(),
            ifop_regex: Regex::new(r"^\s*#\s*if\s*([\w|\d|_]+)\s*([^\s]*)\s*([-\w|\d]+)").unwrap(),
            else_ifdef_regex: Regex::new(r"^\s*#\s*else\s+ifdef\s*([\w|\d|_]+)").unwrap(),
            else_regex: Regex::new(r"^\s*#\s*else").unwrap(),
            endif_regex: Regex::new(r"^\s*#\s*endif").unwrap(),
            // define_regex: Regex::new(r"^\s*#\s*define\s+([\w|\d|_]+)\s*([-\w|\d]+)?").unwrap(),
            define_regex: Regex::new(r"^\s*#\s*define\s+(\w+)\s*(-?\w+\.\d+|-?\w+)?").unwrap(),
            def_regex: Regex::new(r"#\s*([\w|\d|_]+)").unwrap(),
            def_regex_delimited: Regex::new(r"#\s*\{([\w|\d|_]+)\}").unwrap(),
        }
    }
}

struct Scope {
    // Is the current scope one in which we should accept new lines into the output?
    accepting_lines: bool,

    // Has this scope ever accepted lines?
    // Needs to be tracked for #else ifdef chains.
    has_accepted_lines: bool,
}

impl Scope {
    fn new(should_lines_be_accepted: bool) -> Self {
        Self {
            accepting_lines: should_lines_be_accepted,
            has_accepted_lines: should_lines_be_accepted,
        }
    }

    fn is_accepting_lines(&self) -> bool {
        self.accepting_lines
    }

    fn stop_accepting_lines(&mut self) {
        self.accepting_lines = false;
    }

    fn start_accepting_lines_if_appropriate(&mut self) {
        if !self.has_accepted_lines {
            self.has_accepted_lines = true;
            self.accepting_lines = true;
        } else {
            self.accepting_lines = false;
        }
    }
}

fn add_directory<P: AsRef<Path>>(
    shaders: &mut HashMap<ShaderImport, Shader>,
    path: P,
    full_path: bool,
) -> Result<(), std::io::Error>{
    for file in fs::read_dir(path)?{
        let file = file?;
        if !file.metadata()?.is_file(){
            continue
        }
        let file = file.path();
        let Some(ext) = file.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase()) else { continue };
        if ext != "wgsl"{
            continue
        }

        let Ok(source_string) = std::fs::read_to_string(&file) else { continue };
        let file = if !full_path{
            file.file_stem().unwrap().to_os_string()
        } else {
            file.into_os_string()
        };
        let Ok(file) = file.into_string() else { continue };
        let shaderimport = ShaderImport::Custom(file);
        let shader = Shader::from_wgsl(source_string);
        shaders.insert(shaderimport, shader);
    }
    Ok(())
}

impl ShaderProcessor {
    pub fn process(
        &self,
        shader: &Shader,
        shader_defs: &[ShaderDefVal],
        shaders: &HashMap<ShaderImport, Shader>,
        // import_handles: &HashMap<ShaderImport, Handle<Shader>>,
    ) -> Result<ProcessedShader, ProcessShaderError> {
        let mut shader_defs_unique =
            HashMap::<String, ShaderDefVal>::from_iter(shader_defs.iter().map(|v| match v {
                ShaderDefVal::Bool(k, _)
                | ShaderDefVal::Int(k, _)
                | ShaderDefVal::UInt(k, _)
                | ShaderDefVal::Float(k, _)
                | ShaderDefVal::Any(k, _) => {
                    (k.clone(), v.clone())
                }
            }));
        self.process_inner(shader, &mut shader_defs_unique, shaders).clone()
    }
    
    fn process_inner<'a>(
        &self,
        shader: &Shader,
        shader_defs_unique: &mut HashMap<String, ShaderDefVal>,
        shaders: &HashMap<ShaderImport, Shader>,
        // final_output: &'a mut HashMap<ShaderImport, Result<ProcessedShader, ProcessShaderError>>,
        // import_handles: &HashMap<ShaderImport, Handle<Shader>>,
    ) -> Result<ProcessedShader, ProcessShaderError> {
        
        // let shader_str = match &shader.source {
        //     Source::Wgsl(source) => source.deref(),
        //     Source::Glsl(source, _stage) => source.deref(),
        //     Source::SpirV(source) => {
        //         if shader_defs_unique.is_empty() {
        //             return Ok(ProcessedShader::SpirV(source.clone()));
        //         }
        //         return Err(ProcessShaderError::ShaderFormatDoesNotSupportShaderDefs);
        //     }
        // };

        let shader_str: &str = &shader.source.0;
        let mut scopes = vec![Scope::new(true)];
        let mut final_string = String::new();
        for line in shader_str.lines() {
            if let Some(cap) = self.ifdef_regex.captures(line) {
                let def = cap.get(1).unwrap();

                let current_valid = scopes.last().unwrap().is_accepting_lines();
                let has_define = shader_defs_unique.contains_key(def.as_str());

                scopes.push(Scope::new(current_valid && has_define));
            } else if let Some(cap) = self.ifndef_regex.captures(line) {
                let def = cap.get(1).unwrap();

                let current_valid = scopes.last().unwrap().is_accepting_lines();
                let has_define = shader_defs_unique.contains_key(def.as_str());

                scopes.push(Scope::new(current_valid && !has_define));
            } else if let Some(cap) = self.ifop_regex.captures(line) {
                let def = cap.get(1).unwrap();
                let op = cap.get(2).unwrap();
                let val = cap.get(3).unwrap();

                fn act_on<T: PartialEq + PartialOrd>(a: T, b: T, op: &str) -> Result<bool, ProcessShaderError> {
                    match op {
                        "==" => Ok(a == b),
                        "!=" => Ok(a != b),
                        ">" => Ok(a > b),
                        ">=" => Ok(a >= b),
                        "<" => Ok(a < b),
                        "<=" => Ok(a <= b),
                        _ => Err(ProcessShaderError::UnknownShaderDefOperator {
                            operator: op.to_string(),
                        }),
                    }
                }

                let def = shader_defs_unique.get(def.as_str()).ok_or(
                    ProcessShaderError::UnknownShaderDef {
                        shader_def_name: def.as_str().to_string(),
                    },
                )?;
                let new_scope = match def {
                    ShaderDefVal::Bool(name, def) => {
                        let val = val.as_str().parse().map_err(|_| {
                            ProcessShaderError::InvalidShaderDefComparisonValue {
                                shader_def_name: name.clone(),
                                value: val.as_str().to_string(),
                                expected: "bool".to_string(),
                            }
                        })?;
                        act_on(*def, val, op.as_str())?
                    }
                    ShaderDefVal::Int(name, def) => {
                        let val = val.as_str().parse().map_err(|_| {
                            ProcessShaderError::InvalidShaderDefComparisonValue {
                                shader_def_name: name.clone(),
                                value: val.as_str().to_string(),
                                expected: "int".to_string(),
                            }
                        })?;
                        act_on(*def, val, op.as_str())?
                    }
                    ShaderDefVal::UInt(name, def) => {
                        let val = val.as_str().parse().map_err(|_| {
                            ProcessShaderError::InvalidShaderDefComparisonValue {
                                shader_def_name: name.clone(),
                                value: val.as_str().to_string(),
                                expected: "uint".to_string(),
                            }
                        })?;
                        act_on(*def, val, op.as_str())?
                    }
                    ShaderDefVal::Float(name, def) => {
                        let val = val.as_str().parse().map_err(|_| {
                            ProcessShaderError::InvalidShaderDefComparisonValue {
                                shader_def_name: name.clone(),
                                value: val.as_str().to_string(),
                                expected: "float".to_string(),
                            }
                        })?;
                        act_on(*def, val, op.as_str())?
                    }
                    ShaderDefVal::Any(name, def) => {
                        let op_str = op.as_str();
                        if !((op_str == "==") | (op_str == "!=")){
                            return Err(ProcessShaderError::InvalidShaderDefComparisonAny{
                                shader_def_name: name.to_string(),
                                value: def.as_str().to_string(),
                            })
                        }
                        act_on(def.as_str(), val.as_str(), op_str)?
                    }
                };

                let current_valid = scopes.last().unwrap().is_accepting_lines();

                scopes.push(Scope::new(current_valid && new_scope));
            } else if let Some(cap) = self.else_ifdef_regex.captures(line) {
                // When should we accept the code in an
                //
                //  #else ifdef FOO
                //      <stuff>
                //  #endif
                //
                // block? Conditions:
                //  1. The parent scope is accepting lines.
                //  2. The current scope is _not_ accepting lines.
                //  3. FOO is defined.
                //  4. We haven't already accepted another #ifdef (or #else ifdef) in the current scope.

                // Condition 1
                let mut parent_accepting = true;

                if scopes.len() > 1 {
                    parent_accepting = scopes[scopes.len() - 2].is_accepting_lines();
                }

                if let Some(current) = scopes.last_mut() {
                    // Condition 2
                    let current_accepting = current.is_accepting_lines();

                    // Condition 3
                    let def = cap.get(1).unwrap();
                    let has_define = shader_defs_unique.contains_key(def.as_str());

                    if parent_accepting && !current_accepting && has_define {
                        // Condition 4: Enforced by [`Scope`].
                        current.start_accepting_lines_if_appropriate();
                    } else {
                        current.stop_accepting_lines();
                    }
                }
            } else if self.else_regex.is_match(line) {
                let mut parent_accepting = true;

                if scopes.len() > 1 {
                    parent_accepting = scopes[scopes.len() - 2].is_accepting_lines();
                }
                if let Some(current) = scopes.last_mut() {
                    // Using #else means that we only want to accept those lines in the output
                    // if the stuff before #else was _not_ accepted.
                    // That's why we stop accepting here if we were currently accepting.
                    //
                    // Why do we care about the parent scope?
                    // Because if we have something like this:
                    //
                    //  #ifdef NOT_DEFINED
                    //      // Not accepting lines
                    //      #ifdef NOT_DEFINED_EITHER
                    //          // Not accepting lines
                    //      #else
                    //          // This is now accepting lines relative to NOT_DEFINED_EITHER
                    //          <stuff>
                    //      #endif
                    //  #endif
                    //
                    // We don't want to actually add <stuff>.

                    if current.is_accepting_lines() || !parent_accepting {
                        current.stop_accepting_lines();
                    } else {
                        current.start_accepting_lines_if_appropriate();
                    }
                }
            } else if self.endif_regex.is_match(line) {
                scopes.pop();
                if scopes.is_empty() {
                    // if let (Some(results), Some(name)) = (final_output, shader_name){
                    //     results.insert(name.clone(), Err(ProcessShaderError::TooManyEndIfs));
                    // }
                    return Err(ProcessShaderError::TooManyEndIfs);
                }
            } else if scopes.last().unwrap().is_accepting_lines() {
                if let Some(cap) = SHADER_IMPORT_PROCESSOR
                    .import_asset_path_regex
                    .captures(line)
                {
                    let import = ShaderImport::AssetPath(cap.get(1).unwrap().as_str().to_string());
                    self.apply_import(
                        // import_handles,
                        shaders,
                        &import,
                        // shader,
                        shader_defs_unique,
                        &mut final_string,
                    )?;
                } else if let Some(cap) = SHADER_IMPORT_PROCESSOR
                    .import_custom_path_regex
                    .captures(line)
                {
                    let import = ShaderImport::Custom(cap.get(1).unwrap().as_str().to_string());
                    self.apply_import(
                        // import_handles,
                        shaders,
                        &import,
                        // shader,
                        shader_defs_unique,
                        &mut final_string,
                    )?;
                } else if SHADER_IMPORT_PROCESSOR
                    .define_import_path_regex
                    .is_match(line)
                {
                    // ignore import path lines
                } else if let Some(cap) = self.define_regex.captures(line) {
                    let def = cap.get(1).unwrap();
                    let name = def.as_str().to_string();

                    if let Some(val) = cap.get(2) {
                        if let Ok(val) = val.as_str().parse::<u32>() {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::UInt(name, val));
                        } else if let Ok(val) = val.as_str().parse::<i32>() {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::Int(name, val));
                        } else if let Ok(val) = val.as_str().parse::<bool>() {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::Bool(name, val));
                        } else if let Ok(val) = val.as_str().parse::<f32>() {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::Float(name, val));
                        } else {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::Any(name, val.as_str().to_string()));
                        }
                    } else {
                        shader_defs_unique.insert(name.clone(), ShaderDefVal::Bool(name, true));
                    }
                } else {
                    let mut line_with_defs = line.to_string();
                    for capture in self.def_regex.captures_iter(line) {
                        let def = capture.get(1).unwrap();
                        if let Some(def) = shader_defs_unique.get(def.as_str()) {
                            line_with_defs = self
                                .def_regex
                                .replace(&line_with_defs, def.value_as_string())
                                .to_string();
                        }
                    }
                    for capture in self.def_regex_delimited.captures_iter(line) {
                        let def = capture.get(1).unwrap();
                        if let Some(def) = shader_defs_unique.get(def.as_str()) {
                            line_with_defs = self
                                .def_regex_delimited
                                .replace(&line_with_defs, def.value_as_string())
                                .to_string();
                        }
                    }
                    final_string.push_str(&line_with_defs);
                    final_string.push('\n');
                }
            }
        }

        if scopes.len() != 1 {
            return Err(ProcessShaderError::NotEnoughEndIfs)
        }

        let processed_source = Cow::from(final_string);

        // match &shader.source {
            
        //     Source::Wgsl(_source) => Ok(ProcessedShader::Wgsl(processed_source)),
        //     Source::Glsl(_source, stage) => Ok(ProcessedShader::Glsl(processed_source, *stage)),
        //     Source::SpirV(_source) => {
        //         unreachable!("SpirV has early return");
        //     }
        // }
        Ok(ProcessedShader(processed_source))
        // final_output.insert(shader_name.clone(), Ok(ProcessedShader(processed_source)));
        // final_output.get(shader_name).unwrap() 
    }

    fn apply_import(
        &self,
        // import_handles: &HashMap<ShaderImport, Handle<Shader>>,
        shaders: &HashMap<ShaderImport, Shader>,
        import: &ShaderImport,
        // shader: &Shader,
        shader_defs_unique: &mut HashMap<String, ShaderDefVal>,
        final_string: &mut String,
        // final_output: &mut HashMap<ShaderImport, Result<ProcessedShader, ProcessShaderError>>,
    ) -> Result<(), ProcessShaderError> {
        let imported_shader = shaders
            .get(import)
            .ok_or_else(|| ProcessShaderError::UnresolvedImport(import.clone()))?;

        // let imported_shader = match imported_shader{
        //     Ok(val) => val,
        //     Err(err) => return Err(err)
        // };
        let imported_processed =
            self.process_inner(imported_shader, shader_defs_unique, shaders);
        
        let imported_processed = match imported_processed{
            Ok(val) => val,
            Err(err) => return Err(err.clone()),
        };


        final_string.push_str(&imported_processed.0);
        
        // match &shader.source {
        //     Source::Wgsl(_) => {
        //         if let ProcessedShader::Wgsl(import_source) = &imported_processed {
        //             final_string.push_str(import_source);
        //         } else {
        //             return Err(ProcessShaderError::MismatchedImportFormat(import.clone()));
        //         }
        //     }
        //     Source::Glsl(_, _) => {
        //         if let ProcessedShader::Glsl(import_source, _) = &imported_processed {
        //             final_string.push_str(import_source);
        //         } else {
        //             return Err(ProcessShaderError::MismatchedImportFormat(import.clone()));
        //         }
        //     }
        //     Source::SpirV(_) => {
        //         return Err(ProcessShaderError::ShaderFormatDoesNotSupportImports);
        //     }
        // }

        Ok(())
    }
}



pub mod tests{
    use super::*;

    #[test]
    fn test_read_dir(){
        // let mut map = Default::default();
        // add_directory(&mut map, ".");
        // dbg!(map);
        
        let mut map = Default::default();
        dbg!(add_directory(&mut map, r"src\test_shaders", false));
        dbg!(&map);
        let processor = ShaderProcessor::default();

        for (_, shader) in map.iter(){
            let idk = processor.process(shader, &[], &map);
            dbg!(idk.unwrap());
        }
        // processor.process(shader, , )
    }
}
