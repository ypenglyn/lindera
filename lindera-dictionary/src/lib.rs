use byteorder::LittleEndian;
use byteorder::WriteBytesExt;
use lindera_core::file_util::read_utf8_file;
use lindera_core::user_dictionary::UserDictionary;
use lindera_core::word_entry::WordEntry;
use lindera_core::word_entry::WordId;
use log::info;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use lindera_core::character_definition::CharacterDefinitions;
use lindera_core::connection::ConnectionCostMatrix;
use lindera_core::dictionary::Dictionary;
use lindera_core::error::LinderaErrorKind;
use lindera_core::prefix_dict::PrefixDict;
use lindera_core::unknown_dictionary::UnknownDictionary;
use lindera_core::LinderaResult;
use yada::builder::DoubleArrayBuilder;
use yada::DoubleArray;

fn read_file(path: PathBuf) -> LinderaResult<Vec<u8>> {
    fs::read(path).map_err(|e| LinderaErrorKind::Io.with_error(e))
}

pub fn load_dictionary(path: PathBuf) -> LinderaResult<Dictionary> {
    Ok(Dictionary {
        dict: prefix_dict(path.clone())?,
        cost_matrix: connection(path.clone())?,
        char_definitions: char_def(path.clone())?,
        unknown_dictionary: unknown_dict(path.clone())?,
        words_idx_data: words_idx_data(path.clone())?,
        words_data: words_data(path)?,
    })
}

pub fn char_def(dir: PathBuf) -> LinderaResult<CharacterDefinitions> {
    let path = dir.join("char_def.bin");
    let data = read_file(path)?;

    CharacterDefinitions::load(data.as_slice())
}

pub fn connection(dir: PathBuf) -> LinderaResult<ConnectionCostMatrix> {
    let path = dir.join("matrix.mtx");
    let data = read_file(path)?;

    Ok(ConnectionCostMatrix::load(data.as_slice()))
}

pub fn prefix_dict(dir: PathBuf) -> LinderaResult<PrefixDict> {
    let unidic_data_path = dir.join("dict.da");
    let unidic_data = read_file(unidic_data_path)?;

    let unidic_vals_path = dir.join("dict.vals");
    let unidic_vals = read_file(unidic_vals_path)?;

    Ok(PrefixDict::from_static_slice(
        unidic_data.as_slice(),
        unidic_vals.as_slice(),
    ))
}

pub fn unknown_dict(dir: PathBuf) -> LinderaResult<UnknownDictionary> {
    let path = dir.join("unk.bin");
    let data = read_file(path)?;

    UnknownDictionary::load(data.as_slice())
}

pub fn words_idx_data(dir: PathBuf) -> LinderaResult<Vec<u8>> {
    let path = dir.join("dict.wordsidx");
    read_file(path)
}

pub fn words_data(dir: PathBuf) -> LinderaResult<Vec<u8>> {
    let path = dir.join("dict.words");
    read_file(path)
}

#[derive(Debug)]
pub struct CsvRow<'a> {
    surface_form: &'a str,
    left_id: u32,
    #[allow(dead_code)]
    right_id: u32,
    word_cost: i32,

    pos_level1: &'a str,
    pos_level2: &'a str,
    pos_level3: &'a str,
    pos_level4: &'a str,

    pub conjugation_type: &'a str,
    pub conjugate_form: &'a str,

    pub base_form: &'a str,
    pub reading: &'a str,
    pronunciation: &'a str,
}

impl<'a> CsvRow<'a> {
    const SIMPLE_USERDIC_FIELDS_NUM: usize = 3;
    const DETAILED_USERDIC_FIELDS_NUM: usize = 13;

    fn from_line(line: &'a str) -> LinderaResult<CsvRow<'a>> {
        let fields: Vec<_> = line.split(',').collect();

        Ok(CsvRow {
            surface_form: fields[0],
            left_id: u32::from_str(fields[1]).map_err(|_err| {
                LinderaErrorKind::Parse.with_error(anyhow::anyhow!("failed to parse left_id"))
            })?,
            right_id: u32::from_str(fields[2]).map_err(|_err| {
                LinderaErrorKind::Parse.with_error(anyhow::anyhow!("failed to parse right_id"))
            })?,
            word_cost: i32::from_str(fields[3]).map_err(|_err| {
                LinderaErrorKind::Parse.with_error(anyhow::anyhow!("failed to parse word_cost"))
            })?,

            pos_level1: fields[4],
            pos_level2: fields[5],
            pos_level3: fields[6],
            pos_level4: fields[7],

            conjugation_type: fields[8],
            conjugate_form: fields[9],

            base_form: fields[10],
            reading: fields[11],
            pronunciation: fields[12],
        })
    }

    fn from_line_user_dict(line: &'a str) -> LinderaResult<CsvRow<'a>> {
        let fields: Vec<_> = line.split(',').collect();

        match fields.len() {
            Self::SIMPLE_USERDIC_FIELDS_NUM => Ok(CsvRow {
                surface_form: fields[0],
                left_id: 0,
                right_id: 0,
                word_cost: -10000,

                pos_level1: fields[1],
                pos_level2: "*",
                pos_level3: "*",
                pos_level4: "*",

                conjugation_type: "*",
                conjugate_form: "*",

                base_form: fields[0],
                reading: fields[2],
                pronunciation: "*",
            }),
            Self::DETAILED_USERDIC_FIELDS_NUM => CsvRow::from_line(line),
            _ => Err(LinderaErrorKind::Content.with_error(anyhow::anyhow!(
                "user dictionary should be a CSV with {} or {} fields",
                Self::SIMPLE_USERDIC_FIELDS_NUM,
                Self::DETAILED_USERDIC_FIELDS_NUM
            ))),
        }
    }
}

pub fn build_user_dict(input_file: &Path) -> LinderaResult<UserDictionary> {
    info!("reading {:?}", input_file);
    let data: String = read_utf8_file(input_file)?;

    let lines: Vec<&str> = data.lines().collect();
    let mut rows: Vec<CsvRow> = lines
        .iter()
        .map(|line| CsvRow::from_line_user_dict(line))
        .collect::<Result<_, _>>()
        .map_err(|err| LinderaErrorKind::Parse.with_error(anyhow::anyhow!(err)))?;

    // sorting entries
    rows.sort_by_key(|row| row.surface_form);

    let mut word_entry_map: BTreeMap<String, Vec<WordEntry>> = BTreeMap::new();

    for (row_id, row) in rows.iter().enumerate() {
        word_entry_map
            .entry(row.surface_form.to_string())
            .or_insert_with(Vec::new)
            .push(WordEntry {
                word_id: WordId(row_id as u32, false),
                word_cost: row.word_cost as i16,
                cost_id: row.left_id as u16,
            });
    }

    let mut words_data = Vec::<u8>::new();
    let mut words_idx_data = Vec::<u8>::new();
    for row in rows.iter() {
        let word = vec![
            row.pos_level1.to_string(),
            row.pos_level2.to_string(),
            row.pos_level3.to_string(),
            row.pos_level4.to_string(),
            row.conjugation_type.to_string(),
            row.conjugate_form.to_string(),
            row.base_form.to_string(),
            row.reading.to_string(),
            row.pronunciation.to_string(),
        ];
        let offset = words_data.len();
        words_idx_data
            .write_u32::<LittleEndian>(offset as u32)
            .map_err(|err| LinderaErrorKind::Io.with_error(anyhow::anyhow!(err)))?;
        bincode::serialize_into(&mut words_data, &word)
            .map_err(|err| LinderaErrorKind::Serialize.with_error(anyhow::anyhow!(err)))?;
    }

    let mut id = 0u32;

    // building da
    let mut keyset: Vec<(&[u8], u32)> = vec![];
    for (key, word_entries) in &word_entry_map {
        let len = word_entries.len() as u32;
        assert!(
            len < (1 << 5),
            "{} is {} length. Too long. [{}]",
            key,
            len,
            (1 << 5)
        );
        let val = (id << 5) | len;
        keyset.push((key.as_bytes(), val));
        id += len;
    }

    let da_bytes = DoubleArrayBuilder::build(&keyset).ok_or_else(|| {
        LinderaErrorKind::Io.with_error(anyhow::anyhow!("DoubleArray build error for user dict."))
    })?;

    // building values
    let mut vals_data = Vec::<u8>::new();
    for word_entries in word_entry_map.values() {
        for word_entry in word_entries {
            word_entry
                .serialize(&mut vals_data)
                .map_err(|err| LinderaErrorKind::Serialize.with_error(anyhow::anyhow!(err)))?;
        }
    }

    let dict = PrefixDict {
        da: DoubleArray::new(da_bytes),
        vals_data,
        is_system: false,
    };

    Ok(UserDictionary {
        dict,
        words_idx_data,
        words_data,
    })
}
