use crate::polyhedron::{PolyhedronOptions, PolyhedronOutput};
use calculo::num::{CoerceFrom, Epsilon, Num, Rat};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{DualRepresentation, Row};

pub use crate::lp::{LpBasisStatusIssue, LpBasisStatusResult};
pub use crate::polyhedron::repair::{
    FacetGraphRepairDiagnostics, FacetGraphRepairError, FacetGraphRepairOptions,
    FacetGraphRepairReport, FacetGraphRepairResult, FrontierRepairMode, FrontierRepairReport,
    GeneralFrontierRepairDiagnostics, GeneralFrontierRepairReport, RepairedFacet, SimplicialFacet,
    SimplicialFrontierRepairDiagnostics, SimplicialFrontierRepairError,
    SimplicialFrontierRepairOptions, SimplicialFrontierRepairReport,
    SimplicialFrontierRepairResult,
};
pub use crate::polyhedron::{
    PartialResolveIssue, PartialResolveResult, PreparedPartialResolveMinimal,
    PreparedPartialResolveResult, ResolveError, ResolveOptions,
};

#[derive(Clone, Debug)]
pub struct SimplicialFrontierRidge {
    ridge: Vec<Row>,
    incident_facet: usize,
    dropped_vertex: Row,
}

impl SimplicialFrontierRidge {
    pub fn ridge(&self) -> &[Row] {
        &self.ridge
    }

    pub fn incident_facet(&self) -> usize {
        self.incident_facet
    }

    pub fn dropped_vertex(&self) -> Row {
        self.dropped_vertex
    }
}

pub fn simplicial_frontier_ridge_count(facets: &[Vec<Row>], facet_dimension: usize) -> usize {
    crate::polyhedron::repair::simplicial_frontier_ridge_count(facets, facet_dimension)
}

pub fn simplicial_frontier_ridges(
    facets: &[Vec<Row>],
    facet_dimension: usize,
) -> Vec<SimplicialFrontierRidge> {
    crate::polyhedron::repair::simplicial_frontier_ridges_by_facet_index(facets, facet_dimension)
        .into_iter()
        .map(
            |(ridge, (facet_idx, dropped_vertex))| SimplicialFrontierRidge {
                ridge,
                incident_facet: facet_idx,
                dropped_vertex,
            },
        )
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct Certificate<'a, N: Num, R: DualRepresentation> {
    poly: &'a PolyhedronOutput<N, R>,
    incidence: &'a SetFamily,
}

#[derive(Clone, Copy, Debug)]
pub enum CertificateError {
    MissingOutputIncidence,
}

impl std::error::Error for CertificateError {}

impl std::fmt::Display for CertificateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingOutputIncidence => write!(f, "missing output incidence"),
        }
    }
}

impl<N: Num, R: DualRepresentation> PolyhedronOutput<N, R> {
    pub fn certificate(&self) -> Result<Certificate<'_, N, R>, CertificateError> {
        let Some(incidence) = self.incidence() else {
            return Err(CertificateError::MissingOutputIncidence);
        };
        Ok(Certificate {
            poly: self,
            incidence,
        })
    }
}

impl<'a, N: Num, R: DualRepresentation> Certificate<'a, N, R> {
    pub fn polyhedron(&self) -> &'a PolyhedronOutput<N, R> {
        self.poly
    }

    pub fn resolve_as<M>(
        &self,
        poly_options: PolyhedronOptions,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<PolyhedronOutput<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        self.poly
            .resolve_from_incidence_certificate_as(&poly_options, self.incidence, options, eps)
    }

    pub fn resolve_partial_as<M>(
        &self,
        poly_options: PolyhedronOptions,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<PartialResolveResult<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        self.poly.resolve_partial_from_incidence_certificate_as(
            &poly_options,
            self.incidence,
            options,
            eps,
        )
    }

    pub fn resolve_partial_prepared_as<M>(
        &self,
        poly_options: PolyhedronOptions,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<PreparedPartialResolveResult<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        self.poly
            .resolve_partial_from_incidence_certificate_as_prepared(
                &poly_options,
                self.incidence,
                options,
                eps,
            )
    }

    pub fn resolve_partial_prepared_minimal_as<M>(
        &self,
        poly_options: PolyhedronOptions,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<PreparedPartialResolveMinimal<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        self.poly
            .resolve_partial_from_incidence_certificate_as_prepared_minimal(
                &poly_options,
                self.incidence,
                options,
                eps,
            )
    }
}
