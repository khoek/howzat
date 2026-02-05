use std::cell::Cell;
use std::ptr;
use std::sync::OnceLock;

use rug::Integer;

use ppl_sys as sys;

use crate::{PplError, PplResult, VertexExact};

static PPL_INIT: OnceLock<Result<(), String>> = OnceLock::new();

thread_local! {
    static THREAD_INIT: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn ensure_initialized() -> Result<(), String> {
    PPL_INIT
        .get_or_init(|| {
            let code = unsafe { sys::ppl_initialize() };
            if code == 0 || code == sys::ppl_enum_error_code_PPL_ERROR_INVALID_ARGUMENT {
                return Ok(());
            }
            Err(format!("ppl_initialize failed (code={code})"))
        })
        .clone()
        .and_then(|_| ensure_thread_initialized())
}

fn ensure_thread_initialized() -> Result<(), String> {
    THREAD_INIT.with(|done| {
        if done.get() {
            return Ok(());
        }
        let code = unsafe { sys::ppl_thread_initialize() };
        if code == 0 || code == sys::ppl_enum_error_code_PPL_ERROR_INVALID_ARGUMENT {
            done.set(true);
            return Ok(());
        }
        Err(format!("ppl_thread_initialize failed (code={code})"))
    })
}

pub(crate) fn convex_hull_facets_incidence(
    vertices: &[VertexExact],
    dim: usize,
) -> PplResult<(Vec<Vec<usize>>, Vec<Integer>)> {
    ensure_initialized().map_err(PplError::PplCallFailed)?;

    let mut gs = GeneratorSystem::new()?;
    let mut coeff = Coefficient::new()?;

    for vtx in vertices {
        let mut le = LinearExpression::new_with_dimension(dim)?;
        for (var, coord) in vtx.coords_scaled.iter().enumerate() {
            coeff.assign(coord)?;
            le.add_to_coefficient(var, &coeff)?;
        }

        coeff.assign(&vtx.denom)?;
        let generator = Generator::new_point(&le, &coeff)?;
        gs.insert(&generator)?;
    }

    let poly = Polyhedron::from_generators(gs)?;
    let constraints = poly.minimized_constraints()?;

    let mut it = ConstraintSystemConstIterator::new()?;
    let mut end = ConstraintSystemConstIterator::new()?;
    it.begin(constraints)?;
    end.end(constraints)?;

    let mut facet_to_vertex: Vec<Vec<usize>> = Vec::new();
    let mut facet_coeffs: Vec<Integer> = Vec::new();
    let mut coeff_tmp = Coefficient::new()?;
    let mut eval = Integer::new();
    let row_width = dim.saturating_add(1);

    while !it.eq(&end)? {
        let constraint = it.deref()?;
        it.increment()?;

        if constraint.constraint_type()? == sys::ppl_enum_Constraint_Type_PPL_CONSTRAINT_TYPE_EQUAL as i32
        {
            continue;
        }

        let row_start = facet_coeffs.len();
        facet_coeffs.resize_with(row_start + row_width, Integer::new);

        constraint.inhomogeneous_term(&mut coeff_tmp)?;
        coeff_tmp.to_integer(&mut facet_coeffs[row_start])?;

        for var in 0..dim {
            constraint.coefficient(var, &mut coeff_tmp)?;
            coeff_tmp.to_integer(&mut facet_coeffs[row_start + 1 + var])?;
        }

        let row = &facet_coeffs[row_start..row_start + row_width];
        let b = &row[0];
        let coeffs = &row[1..];
        let mut incidence = Vec::new();
        for (idx, vtx) in vertices.iter().enumerate() {
            eval.clone_from(b);
            eval *= &vtx.denom;
            for (a, x) in coeffs.iter().zip(&vtx.coords_scaled) {
                eval += a * x;
            }
            if eval == 0 {
                incidence.push(idx);
            }
        }
        facet_to_vertex.push(incidence);
    }

    debug_assert_eq!(
        facet_coeffs.len(),
        facet_to_vertex.len() * row_width,
        "coefficient matrix length mismatch"
    );
    Ok((facet_to_vertex, facet_coeffs))
}

struct Coefficient(sys::ppl_Coefficient_t);

impl Coefficient {
    fn new() -> PplResult<Self> {
        let mut ptr = ptr::null_mut();
        let code = unsafe { sys::ppl_new_Coefficient(&mut ptr) };
        if code != 0 || ptr.is_null() {
            return Err(PplError::PplCallFailed(format!(
                "ppl_new_Coefficient failed (code={code})"
            )));
        }
        Ok(Self(ptr))
    }

    fn assign(&mut self, value: &Integer) -> PplResult<()> {
        let raw = value.as_raw();
        let code = unsafe { sys::ppl_assign_Coefficient_from_mpz_t(self.0, raw.cast_mut().cast()) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_assign_Coefficient_from_mpz_t failed (code={code})"
            )));
        }
        Ok(())
    }

    fn to_integer(&self, out: &mut Integer) -> PplResult<()> {
        let code = unsafe { sys::ppl_Coefficient_to_mpz_t(self.0, out.as_raw_mut().cast()) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Coefficient_to_mpz_t failed (code={code})"
            )));
        }
        Ok(())
    }
}

impl Drop for Coefficient {
    fn drop(&mut self) {
        unsafe {
            sys::ppl_delete_Coefficient(self.0);
        }
    }
}

struct LinearExpression(sys::ppl_Linear_Expression_t);

impl LinearExpression {
    fn new_with_dimension(dim: usize) -> PplResult<Self> {
        let mut ptr = ptr::null_mut();
        let d: sys::ppl_dimension_type = dim;
        let code = unsafe { sys::ppl_new_Linear_Expression_with_dimension(&mut ptr, d) };
        if code != 0 || ptr.is_null() {
            return Err(PplError::PplCallFailed(format!(
                "ppl_new_Linear_Expression_with_dimension failed (code={code})"
            )));
        }
        Ok(Self(ptr))
    }

    fn add_to_coefficient(&mut self, var: usize, coeff: &Coefficient) -> PplResult<()> {
        let var: sys::ppl_dimension_type = var;
        let code = unsafe { sys::ppl_Linear_Expression_add_to_coefficient(self.0, var, coeff.0) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Linear_Expression_add_to_coefficient failed (code={code})"
            )));
        }
        Ok(())
    }
}

impl Drop for LinearExpression {
    fn drop(&mut self) {
        unsafe {
            sys::ppl_delete_Linear_Expression(self.0);
        }
    }
}

struct Generator(sys::ppl_Generator_t);

impl Generator {
    fn new_point(le: &LinearExpression, divisor: &Coefficient) -> PplResult<Self> {
        let mut ptr = ptr::null_mut();
        let code = unsafe {
            sys::ppl_new_Generator(
                &mut ptr,
                le.0,
                sys::ppl_enum_Generator_Type_PPL_GENERATOR_TYPE_POINT,
                divisor.0,
            )
        };
        if code != 0 || ptr.is_null() {
            return Err(PplError::PplCallFailed(format!(
                "ppl_new_Generator(point) failed (code={code})"
            )));
        }
        Ok(Self(ptr))
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        unsafe {
            sys::ppl_delete_Generator(self.0);
        }
    }
}

struct GeneratorSystem(sys::ppl_Generator_System_t);

impl GeneratorSystem {
    fn new() -> PplResult<Self> {
        let mut ptr = ptr::null_mut();
        let code = unsafe { sys::ppl_new_Generator_System(&mut ptr) };
        if code != 0 || ptr.is_null() {
            return Err(PplError::PplCallFailed(format!(
                "ppl_new_Generator_System failed (code={code})"
            )));
        }
        Ok(Self(ptr))
    }

    fn insert(&mut self, generator: &Generator) -> PplResult<()> {
        let code = unsafe { sys::ppl_Generator_System_insert_Generator(self.0, generator.0) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Generator_System_insert_Generator failed (code={code})"
            )));
        }
        Ok(())
    }
}

impl Drop for GeneratorSystem {
    fn drop(&mut self) {
        unsafe {
            sys::ppl_delete_Generator_System(self.0);
        }
    }
}

struct Polyhedron(sys::ppl_Polyhedron_t);

impl Polyhedron {
    fn from_generators(gs: GeneratorSystem) -> PplResult<Self> {
        let mut ptr = ptr::null_mut();
        let code = unsafe { sys::ppl_new_C_Polyhedron_recycle_Generator_System(&mut ptr, gs.0) };
        if code != 0 || ptr.is_null() {
            return Err(PplError::PplCallFailed(format!(
                "ppl_new_C_Polyhedron_recycle_Generator_System failed (code={code})"
            )));
        }
        Ok(Self(ptr))
    }

    fn minimized_constraints(&self) -> PplResult<sys::ppl_const_Constraint_System_t> {
        let mut cs: sys::ppl_const_Constraint_System_t = ptr::null();
        let code = unsafe { sys::ppl_Polyhedron_get_minimized_constraints(self.0, &mut cs) };
        if code != 0 || cs.is_null() {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Polyhedron_get_minimized_constraints failed (code={code})"
            )));
        }
        Ok(cs)
    }
}

impl Drop for Polyhedron {
    fn drop(&mut self) {
        unsafe {
            sys::ppl_delete_Polyhedron(self.0);
        }
    }
}

struct ConstraintSystemConstIterator(sys::ppl_Constraint_System_const_iterator_t);

impl ConstraintSystemConstIterator {
    fn new() -> PplResult<Self> {
        let mut ptr = ptr::null_mut();
        let code = unsafe { sys::ppl_new_Constraint_System_const_iterator(&mut ptr) };
        if code != 0 || ptr.is_null() {
            return Err(PplError::PplCallFailed(format!(
                "ppl_new_Constraint_System_const_iterator failed (code={code})"
            )));
        }
        Ok(Self(ptr))
    }

    fn begin(&mut self, cs: sys::ppl_const_Constraint_System_t) -> PplResult<()> {
        let code = unsafe { sys::ppl_Constraint_System_begin(cs, self.0) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Constraint_System_begin failed (code={code})"
            )));
        }
        Ok(())
    }

    fn end(&mut self, cs: sys::ppl_const_Constraint_System_t) -> PplResult<()> {
        let code = unsafe { sys::ppl_Constraint_System_end(cs, self.0) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Constraint_System_end failed (code={code})"
            )));
        }
        Ok(())
    }

    fn eq(&self, other: &Self) -> PplResult<bool> {
        let code = unsafe { sys::ppl_Constraint_System_const_iterator_equal_test(self.0, other.0) };
        if code < 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Constraint_System_const_iterator_equal_test failed (code={code})"
            )));
        }
        Ok(code != 0)
    }

    fn increment(&mut self) -> PplResult<()> {
        let code = unsafe { sys::ppl_Constraint_System_const_iterator_increment(self.0) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Constraint_System_const_iterator_increment failed (code={code})"
            )));
        }
        Ok(())
    }

    fn deref(&self) -> PplResult<ConstraintRef> {
        let mut c: sys::ppl_const_Constraint_t = ptr::null();
        let code = unsafe { sys::ppl_Constraint_System_const_iterator_dereference(self.0, &mut c) };
        if code != 0 || c.is_null() {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Constraint_System_const_iterator_dereference failed (code={code})"
            )));
        }
        Ok(ConstraintRef(c))
    }
}

impl Drop for ConstraintSystemConstIterator {
    fn drop(&mut self) {
        unsafe {
            sys::ppl_delete_Constraint_System_const_iterator(self.0);
        }
    }
}

#[derive(Clone, Copy)]
struct ConstraintRef(sys::ppl_const_Constraint_t);

impl ConstraintRef {
    fn constraint_type(&self) -> PplResult<i32> {
        Ok(unsafe { sys::ppl_Constraint_type(self.0) } as i32)
    }

    fn inhomogeneous_term(&self, coeff: &mut Coefficient) -> PplResult<()> {
        let code = unsafe { sys::ppl_Constraint_inhomogeneous_term(self.0, coeff.0) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Constraint_inhomogeneous_term failed (code={code})"
            )));
        }
        Ok(())
    }

    fn coefficient(&self, var: usize, coeff: &mut Coefficient) -> PplResult<()> {
        let var: sys::ppl_dimension_type = var;
        let code = unsafe { sys::ppl_Constraint_coefficient(self.0, var, coeff.0) };
        if code != 0 {
            return Err(PplError::PplCallFailed(format!(
                "ppl_Constraint_coefficient failed (code={code})"
            )));
        }
        Ok(())
    }
}
