use super::{
    BinaryAssociationRecord, BinaryRecord, CombiningRule, IdentifierOption, Parameters, PureRecord,
};
use crate::FeosResult;
use rusqlite::{Connection, ToSql, params_from_iter};
use serde::de::DeserializeOwned;
use std::path::Path;

impl<P: Clone, B: Clone, A: CombiningRule<P> + Clone> Parameters<P, B, A> {
    pub fn from_database<F, S>(
        substances: &[S],
        file: F,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        F: AsRef<Path>,
        S: ToSql,
        P: DeserializeOwned + Clone,
        B: DeserializeOwned + Clone,
        A: DeserializeOwned + Clone,
    {
        let conn = Connection::open(file)?;
        let pure_records = PureRecord::from_database(substances, &conn, identifier_option)?;
        let binary_records = BinaryRecord::from_database(substances, &conn, identifier_option)?;
        Self::new(pure_records, binary_records)
    }
}

impl<M, A> PureRecord<M, A> {
    pub fn from_database<S>(
        substances: &[S],
        connection: &Connection,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Vec<Self>>
    where
        S: ToSql,
        M: DeserializeOwned,
        A: DeserializeOwned,
    {
        let values = (0..substances.len())
            .map(|i| format!("({i},?)"))
            .collect::<Vec<_>>()
            .join(",");

        let query = format!(
            "
            WITH input(idx, ident) AS (
                VALUES {values}
            )
            SELECT pr.pure_record
            FROM input
            JOIN pure_records pr
              ON pr.{identifier_option} = input.ident
            "
        );
        let mut stmt = connection.prepare(&query)?;
        stmt.query_and_then(params_from_iter(substances), |r| {
            Ok(serde_json::from_str(&r.get::<_, String>("pure_record")?)?)
        })?
        .collect()
    }
}

impl<B, A> BinaryRecord<usize, B, A> {
    pub fn from_database<S>(
        substances: &[S],
        connection: &Connection,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Vec<Self>>
    where
        S: ToSql,
        B: DeserializeOwned,
        A: DeserializeOwned,
    {
        let values = (0..substances.len())
            .map(|i| format!("({i},?)"))
            .collect::<Vec<_>>()
            .join(",");

        let query = format!(
            "
            WITH input(idx, ident) AS (
                VALUES {values}
            )
            SELECT i1.idx AS comp1, i2.idx AS comp2, br.model_record, br.association_sites
            FROM binary_records br
            JOIN pure_records p1 ON br.id1 = p1.id
            JOIN pure_records p2 ON br.id2 = p2.id
            JOIN input i1 ON p1.{identifier_option} = i1.ident
            JOIN input i2 ON p2.{identifier_option} = i2.ident
            "
        );
        let mut stmt = connection.prepare(&query)?;
        stmt.query_and_then(params_from_iter(substances), |r| {
            let mut id1: i32 = r.get("comp1")?;
            let mut id2: i32 = r.get("comp2")?;
            let model_record = serde_json::from_str(&r.get::<_, String>("model_record")?)?;
            let mut association_sites: Vec<BinaryAssociationRecord<_>> =
                serde_json::from_str(&r.get::<_, String>("association_sites")?)?;
            if id1 > id2 {
                association_sites
                    .iter_mut()
                    .for_each(|a| std::mem::swap(&mut a.id1, &mut a.id2));
                std::mem::swap(&mut id1, &mut id2);
            };

            Ok(BinaryRecord::with_association(
                id1 as usize,
                id2 as usize,
                model_record,
                association_sites,
            ))
        })?
        .collect()
    }
}
