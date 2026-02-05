use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use syntheon as syn;

const PPL_TAG: &str = "92d0704d3309d55f39a647595f8383b86fcd57e1";

const PERF_FLAGS: &[&str] = &["-O3", "-DNDEBUG", "-g0", "-fomit-frame-pointer"];
const NATIVE_CPU_FLAGS: &[&str] = &["-march=native", "-mtune=native"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoeffMode {
    Gmp,
    I64,
}

impl CoeffMode {
    fn cache_component(self) -> &'static str {
        match self {
            CoeffMode::Gmp => "gmp",
            CoeffMode::I64 => "i64",
        }
    }

    fn configure_arg(self) -> &'static str {
        match self {
            CoeffMode::Gmp => "mpz",
            CoeffMode::I64 => "native-int64",
        }
    }
}

#[derive(Clone)]
struct PplLayout {
    archive_path: PathBuf,
    source_dir: PathBuf,
    build_dir: PathBuf,
    install_dir: PathBuf,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GMP");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_I64");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_USE_SYSTEM_GMP");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PIC");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let use_system_gmp = env::var_os("CARGO_FEATURE_USE_SYSTEM_GMP").is_some();
    let coeff_mode = coeff_mode();
    let needs_pic = env::var_os("CARGO_FEATURE_PIC").is_some();
    let perf_flags = perf_flags_string(needs_pic);
    let out_dir = syn::out_dir();

    let gmpxx_lib_dir = (coeff_mode == CoeffMode::Gmp && !use_system_gmp)
        .then(|| ensure_gmpxx_static(&out_dir, &perf_flags));

    let ppl_layout = ppl_layout(coeff_mode, needs_pic);
    println!("cargo:rerun-if-changed={}", ppl_layout.archive_path.display());
    ensure_ppl_source(&ppl_layout);
    autoreconf_ppl(&ppl_layout.source_dir);

    let install_dir = build_ppl(
        &ppl_layout,
        coeff_mode,
        needs_pic,
        &perf_flags,
        gmpxx_lib_dir.as_deref(),
    );

    let include_dir = install_dir.join("include");
    let lib_dir = ppl_lib_dir(&install_dir);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=ppl_c");
    println!("cargo:rustc-link-lib=static=ppl");

    if coeff_mode == CoeffMode::Gmp {
        if use_system_gmp {
            println!("cargo:rustc-link-lib=gmpxx");
        } else {
            let gmpxx_lib_dir = gmpxx_lib_dir
                .as_ref()
                .expect("ppl-sys: missing gmpxx build dir")
                .clone();
            println!("cargo:rustc-link-search=native={}", gmpxx_lib_dir.display());
            println!("cargo:rustc-link-lib=static=gmpxx");
        }
    }

    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-lib=c++"),
        _ => {
            if env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
                println!("cargo:rustc-link-lib=stdc++");
            }
        }
    }

    if env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }

    generate_bindings(&include_dir, coeff_mode);
}

fn coeff_mode() -> CoeffMode {
    let gmp = env::var_os("CARGO_FEATURE_GMP").is_some();
    let i64 = env::var_os("CARGO_FEATURE_I64").is_some();
    match (gmp, i64) {
        (true, false) => CoeffMode::Gmp,
        (false, true) => CoeffMode::I64,
        (false, false) => panic!("ppl-sys: enable exactly one of: gmp (default), i64"),
        (true, true) => panic!("ppl-sys: enable at most one of: gmp, i64"),
    }
}

fn ppl_layout(coeff_mode: CoeffMode, needs_pic: bool) -> PplLayout {
    let archive_path = syn::vendor_dir().join(format!("ppl-{PPL_TAG}.tar.gz"));
    if !archive_path.is_file() {
        panic!("missing vendored ppl archive at {}", archive_path.display());
    }

    let cache_root = syn::cache_root();
    let dir_key = format!(
        "{}-{}{}",
        syn::sanitize_component(PPL_TAG),
        coeff_mode.cache_component(),
        if needs_pic { "-pic" } else { "" },
    );
    let root = cache_root.join(dir_key);

    PplLayout {
        archive_path,
        source_dir: root.join(format!("PPL-{PPL_TAG}")),
        build_dir: root.join("build"),
        install_dir: root.join("install"),
    }
}

fn ppl_install_is_present(root: &Path) -> bool {
    ppl_lib_dir(root).join("libppl.a").exists() && ppl_lib_dir(root).join("libppl_c.a").exists()
}

fn ppl_lib_dir(install_dir: &Path) -> PathBuf {
    for dir in ["lib", "lib64"] {
        let candidate = install_dir.join(dir);
        if candidate.join("libppl.a").exists() && candidate.join("libppl_c.a").exists() {
            return candidate;
        }
    }
    install_dir.join("lib")
}

fn ensure_ppl_source(layout: &PplLayout) -> PathBuf {
    if layout.source_dir.join("configure.ac").exists() {
        return layout.source_dir.clone();
    }

    if let Some(parent) = layout.source_dir.parent() {
        fs::create_dir_all(parent).expect("failed to create ppl source parent directory");
    }
    let root = layout
        .source_dir
        .parent()
        .unwrap_or_else(|| panic!("missing parent for {}", layout.source_dir.display()));
    syn::extract_tar_gz(&layout.archive_path, root);

    if layout.source_dir.join("configure.ac").exists() {
        return layout.source_dir.clone();
    }
    panic!(
        "ppl source tree not found under {} after extraction",
        layout.source_dir.display()
    );
}

fn autoreconf_ppl(source_dir: &Path) {
    if source_dir.join("configure").exists() {
        return;
    }

    let mut cmd = Command::new("autoreconf");
    cmd.args(["-fi"]).current_dir(source_dir);
    syn::run(
        &mut cmd,
        "ppl autoreconf failed (install autoconf, automake, and libtool)",
    );
}

fn build_ppl(
    layout: &PplLayout,
    coeff_mode: CoeffMode,
    needs_pic: bool,
    perf_flags: &str,
    gmpxx_lib_dir: Option<&Path>,
) -> PathBuf {
    if ppl_install_is_present(&layout.install_dir) {
        return layout.install_dir.clone();
    }

    fs::create_dir_all(&layout.build_dir).expect("failed to create ppl build directory");

    let mut configure = Command::new(layout.source_dir.join("configure"));
    configure
        .arg(format!("--prefix={}", layout.install_dir.display()))
        .arg("--enable-shared=no")
        .arg("--enable-static=yes")
        .arg("--enable-documentation=no")
        .arg("--enable-interfaces=c")
        .arg("--enable-ppl_lcdd=no")
        .arg("--enable-ppl_lpsol=no")
        .arg("--enable-ppl_pips=no")
        .args(needs_pic.then_some("--with-pic"))
        .arg(format!("--enable-coefficients={}", coeff_mode.configure_arg()))
        .current_dir(&layout.build_dir)
        .env("CFLAGS", perf_flags)
        .env("CXXFLAGS", perf_flags)
        .env("ac_cv_path_lt_DD", "/bin/dd")
        .env("lt_cv_truncate_bin", "sed -e 4q");

    if coeff_mode == CoeffMode::Gmp && env::var_os("CARGO_FEATURE_USE_SYSTEM_GMP").is_none() {
        let gmpxx_lib_dir = gmpxx_lib_dir.expect("ppl-sys: missing gmpxx build dir");
        if let Some((gmp_include_dir, gmp_lib_dir)) = gmp_paths() {
            let gmpxx_header_dir = gmpxx_header_dir();
            if !gmpxx_header_dir.join("gmpxx.h").is_file() {
                panic!(
                    "ppl-sys: missing {} (gmpxx.h); expected gmp-mpfr-sys to keep GMP sources \
available (enable cnodelete and disable GMP_MPFR_SYS_CACHE).",
                    gmpxx_header_dir.join("gmpxx.h").display()
                );
            }
            let cppflags = format!(
                "-I{} -I{}",
                gmp_include_dir.display(),
                gmpxx_header_dir.display()
            );
            let ldflags = format!(
                "-L{} -L{}",
                gmp_lib_dir.display(),
                gmpxx_lib_dir.display()
            );
            configure.env("CPPFLAGS", cppflags).env("LDFLAGS", ldflags);
        }
    }
    syn::run(&mut configure, "ppl configure failed");

    let jobs = syn::parallel_jobs();
    let mut make = Command::new("make");
    syn::apply_parallel(&mut make, jobs);
    make.current_dir(&layout.build_dir);
    syn::run(&mut make, "ppl make failed");

    let mut make_install = Command::new("make");
    make_install
        .arg("install")
        .current_dir(&layout.build_dir)
        .env("CMAKE_BUILD_PARALLEL_LEVEL", jobs.to_string());
    syn::apply_parallel(&mut make_install, jobs);
    syn::run(&mut make_install, "ppl make install failed");

    if !ppl_install_is_present(&layout.install_dir) {
        panic!(
            "ppl build did not produce libppl.a + libppl_c.a under {}",
            layout.install_dir.display()
        );
    }

    layout.install_dir.clone()
}

fn perf_flag_string() -> String {
    perf_flags().collect::<Vec<_>>().join(" ")
}

fn perf_flags_string(needs_pic: bool) -> String {
    if needs_pic {
        format!("{} -fPIC", perf_flag_string())
    } else {
        perf_flag_string()
    }
}

fn perf_flags() -> impl Iterator<Item = &'static str> {
    PERF_FLAGS
        .iter()
        .copied()
        .chain(native_cpu_flags().iter().copied())
}

fn native_cpu_flags() -> &'static [&'static str] {
    if syn::wants_native_cpu_flags() {
        NATIVE_CPU_FLAGS
    } else {
        &[]
    }
}

fn generate_bindings(include_dir: &Path, coeff_mode: CoeffMode) {
    let out_dir = syn::out_dir();

    let header = "\
typedef __UINT8_TYPE__ uint8_t;\n\
typedef __UINT64_TYPE__ uint64_t;\n\
#include <stdint.h>\n\
#include <stdio.h>\n\
#include <ppl_c.h>\n";

    let mut builder = bindgen::Builder::default()
        .header_contents("ppl_sys.h", header)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Init.
        .allowlist_function("ppl_initialize")
        .allowlist_function("ppl_finalize")
        .allowlist_function("ppl_thread_initialize")
        .allowlist_function("ppl_thread_finalize")
        // Coefficients.
        .allowlist_function("ppl_new_Coefficient")
        .allowlist_function("ppl_delete_Coefficient")
        .allowlist_function("ppl_assign_Coefficient_from_mpz_t")
        .allowlist_function("ppl_Coefficient_to_mpz_t")
        .allowlist_function("ppl_Coefficient_is_bounded")
        // Linear expressions.
        .allowlist_function("ppl_new_Linear_Expression_with_dimension")
        .allowlist_function("ppl_delete_Linear_Expression")
        .allowlist_function("ppl_Linear_Expression_add_to_coefficient")
        .allowlist_function("ppl_Linear_Expression_add_to_inhomogeneous")
        // Generators and systems.
        .allowlist_function("ppl_new_Generator")
        .allowlist_function("ppl_delete_Generator")
        .allowlist_function("ppl_Generator_type")
        .allowlist_function("ppl_Generator_coefficient")
        .allowlist_function("ppl_Generator_divisor")
        .allowlist_function("ppl_new_Generator_System")
        .allowlist_function("ppl_delete_Generator_System")
        .allowlist_function("ppl_Generator_System_insert_Generator")
        // Polyhedron (V->H conversion).
        .allowlist_function("ppl_new_C_Polyhedron_from_Generator_System")
        .allowlist_function("ppl_new_C_Polyhedron_recycle_Generator_System")
        .allowlist_function("ppl_delete_Polyhedron")
        .allowlist_function("ppl_Polyhedron_get_minimized_constraints")
        // Constraint systems + iteration.
        .allowlist_function("ppl_new_Constraint_System_const_iterator")
        .allowlist_function("ppl_delete_Constraint_System_const_iterator")
        .allowlist_function("ppl_Constraint_System_begin")
        .allowlist_function("ppl_Constraint_System_end")
        .allowlist_function("ppl_Constraint_System_const_iterator_dereference")
        .allowlist_function("ppl_Constraint_System_const_iterator_increment")
        .allowlist_function("ppl_Constraint_System_const_iterator_equal_test")
        // Constraint access.
        .allowlist_function("ppl_Constraint_type")
        .allowlist_function("ppl_Constraint_coefficient")
        .allowlist_function("ppl_Constraint_inhomogeneous_term")
        // Core types/consts.
        .allowlist_type("ppl_dimension_type")
        .allowlist_type("ppl_Coefficient_t")
        .allowlist_type("ppl_const_Coefficient_t")
        .allowlist_type("ppl_Linear_Expression_t")
        .allowlist_type("ppl_const_Linear_Expression_t")
        .allowlist_type("ppl_Generator_t")
        .allowlist_type("ppl_const_Generator_t")
        .allowlist_type("ppl_Generator_System_t")
        .allowlist_type("ppl_const_Generator_System_t")
        .allowlist_type("ppl_Polyhedron_t")
        .allowlist_type("ppl_const_Polyhedron_t")
        .allowlist_type("ppl_const_Constraint_System_t")
        .allowlist_type("ppl_Constraint_System_const_iterator_t")
        .allowlist_type("ppl_const_Constraint_System_const_iterator_t")
        .allowlist_type("ppl_const_Constraint_t")
        .allowlist_var("PPL_GENERATOR_TYPE_.*")
        .allowlist_var("PPL_CONSTRAINT_TYPE_.*")
        .allowlist_var("PPL_ERROR_.*");

    builder = builder
        .allowlist_type("ppl_enum_error_code")
        .allowlist_type("ppl_enum_Constraint_Type")
        .allowlist_type("ppl_enum_Generator_Type");

    builder = builder
        .clang_arg("-x")
        .clang_arg("c")
        .clang_arg("-std=gnu99")
        .clang_arg(format!("--target={}", syn::target_triple()));

    for arg in syn::clang_system_include_args() {
        builder = builder.clang_arg(arg);
    }
    for arg in syn::clang_macos_sysroot_args() {
        builder = builder.clang_arg(arg);
    }

    builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    if coeff_mode == CoeffMode::Gmp {
        if let Some((gmp_include_dir, _)) = gmp_paths() {
            builder = builder.clang_arg(format!("-I{}", gmp_include_dir.display()));
        }
    }

    let bindings = builder
        .generate()
        .expect("Unable to generate PPL bindings with bindgen");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn gmp_paths() -> Option<(PathBuf, PathBuf)> {
    let include = PathBuf::from(env::var_os("DEP_GMP_INCLUDE_DIR")?);
    let lib = PathBuf::from(env::var_os("DEP_GMP_LIB_DIR")?);

    if include.join("gmp.h").is_file() && lib.exists() {
        Some((include, lib))
    } else {
        None
    }
}

fn gmp_out_dir() -> PathBuf {
    PathBuf::from(env::var("DEP_GMP_OUT_DIR").expect("DEP_GMP_OUT_DIR must be set when using GMP"))
}

fn gmpxx_header_dir() -> PathBuf {
    gmp_out_dir().join("build").join("gmp-src")
}

fn ensure_gmpxx_static(out_dir: &Path, perf_flags: &str) -> PathBuf {
    let gmp_out_dir = gmp_out_dir();
    let gmp_build_dir = gmp_out_dir.join("build").join("gmp-build");
    let gmp_src_dir = gmp_out_dir.join("build").join("gmp-src");
    let cxx_src_dir = gmp_src_dir.join("cxx");
    if !cxx_src_dir.is_dir() {
        panic!(
            "ppl-sys: missing GMP C++ sources at {} (expected gmp-mpfr-sys to build from source)",
            cxx_src_dir.display()
        );
    }
    if !gmp_build_dir.join("config.h").is_file() {
        panic!(
            "ppl-sys: missing GMP build configuration at {} (expected gmp-mpfr-sys to keep build dir)",
            gmp_build_dir.display()
        );
    }

    let build_dir = out_dir.join("gmpxx");
    fs::create_dir_all(&build_dir).expect("ppl-sys: failed to create gmpxx build dir");
    let lib_path = build_dir.join("libgmpxx.a");
    if lib_path.is_file() {
        return build_dir;
    }

    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());

    let mut sources: Vec<PathBuf> = fs::read_dir(&cxx_src_dir)
        .unwrap_or_else(|e| panic!("ppl-sys: failed to read {}: {e}", cxx_src_dir.display()))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let is_cc = path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext == "cc");
            is_cc.then_some(path)
        })
        .collect();
    sources.sort();

    if sources.is_empty() {
        panic!(
            "ppl-sys: no GMP C++ sources found under {}",
            cxx_src_dir.display()
        );
    }

    let mut objects: Vec<PathBuf> = Vec::with_capacity(sources.len());
    for src in sources {
        let stem = src
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("ppl-sys: invalid GMP C++ source path {}", src.display()));
        let obj = build_dir.join(format!("{stem}.o"));

        let mut cmd = Command::new(&cxx);
        cmd.args(perf_flags.split_whitespace())
            .arg("-std=gnu++11")
            .arg(format!("-I{}", gmp_build_dir.display()))
            .arg(format!("-I{}", gmp_src_dir.display()))
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj);
        syn::run(&mut cmd, "ppl-sys: GMP C++ library compile failed");
        objects.push(obj);
    }

    let mut cmd = Command::new(&ar);
    cmd.arg("rcs").arg(&lib_path);
    cmd.args(&objects);
    syn::run(&mut cmd, "ppl-sys: GMP C++ library archive failed");

    if !lib_path.is_file() {
        panic!("ppl-sys: GMP C++ library build failed (missing {})", lib_path.display());
    }

    build_dir
}
