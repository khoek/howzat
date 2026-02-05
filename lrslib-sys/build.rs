use std::env;
use std::fs;
use std::path::PathBuf;

use syntheon as syn;

const LRSLIB_TAG: &str = "073a";
const PERF_FLAGS: &[&str] = &["-O3", "-DNDEBUG", "-g0", "-fomit-frame-pointer"];
const NATIVE_CPU_FLAGS: &[&str] = &["-march=native", "-mtune=native"];

#[derive(Clone)]
struct LrsLayout {
    archive_path: PathBuf,
    source_dir: PathBuf,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GMP");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let layout = lrs_layout();
    println!("cargo:rerun-if-changed={}", layout.archive_path.display());

    ensure_lrslib_source(&layout);
    build_lrslib(&layout);
    generate_bindings(&layout);

    if env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-lib=m");
    }
}

fn vendor_dir() -> PathBuf {
    syn::vendor_dir()
}

fn lrs_layout() -> LrsLayout {
    let archive_path = vendor_dir().join(format!("lrslib-{LRSLIB_TAG}.tar.gz"));
    if !archive_path.is_file() {
        panic!(
            "missing vendored lrslib archive at {}",
            archive_path.display()
        );
    }

    let cache_root = cache_root();
    let dir_key = syn::sanitize_component(LRSLIB_TAG);
    let root = cache_root.join(dir_key);

    LrsLayout {
        archive_path,
        source_dir: root.join(format!("lrslib-{LRSLIB_TAG}")),
    }
}

fn ensure_lrslib_source(layout: &LrsLayout) -> PathBuf {
    if layout.source_dir.join("lrslib.c").exists() {
        return layout.source_dir.clone();
    }

    if let Some(parent) = layout.source_dir.parent() {
        fs::create_dir_all(parent).expect("failed to create lrslib source parent directory");
    }

    let root = layout
        .source_dir
        .parent()
        .unwrap_or_else(|| panic!("missing parent for {}", layout.source_dir.display()));

    syn::extract_tar_gz(&layout.archive_path, root);

    if layout.source_dir.join("lrslib.c").exists() {
        return layout.source_dir.clone();
    }

    panic!(
        "lrslib source tree not found under {} after extraction",
        layout.source_dir.display()
    );
}

fn build_lrslib(layout: &LrsLayout) {
    let arith_dir = layout.source_dir.join("lrsarith-011");
    let use_gmp = env::var_os("CARGO_FEATURE_GMP").is_some();

    let mut sources = vec![
        layout.source_dir.join("lrslib.c"),
        layout.source_dir.join("lrsdriver.c"),
    ];
    if use_gmp {
        sources.push(arith_dir.join("lrsgmp.c"));
    } else {
        sources.push(arith_dir.join("lrslong.c"));
    }

    for src in &sources {
        if !src.is_file() {
            panic!("missing lrslib source file {}", src.display());
        }
    }

    let mut build = cc::Build::new();
    build
        // Vendored C code: don't spam downstream builds with warnings.
        .warnings(false)
        .files(&sources)
        .include(&layout.source_dir)
        .include(&arith_dir)
        .flag_if_supported("-std=gnu99")
        // Build lrslib single-threaded: do NOT define PLRS/MPLRS or pass -fopenmp.
        .define("LRS_QUIET", None)
        // lrslib is primarily a CLI tool; it installs process-wide signal handlers
        // (SIGINT/SIGTERM/etc.) unless SIGNALS is defined. We do not want a library
        // to override the host application's signal behavior.
        .define("SIGNALS", None);

    if use_gmp {
        build.define("GMP", None);
        if let Some(include_dir) = gmp_include_dir() {
            build.include(include_dir);
        }
    } else {
        // Exact fixed-width arithmetic (fast, but can overflow on hard instances).
        //
        // We default to 128-bit if the target supports it.
        build.define("LRSLONG", None).define("SAFE", None);
        if env::var("CARGO_CFG_TARGET_POINTER_WIDTH").as_deref() == Ok("64") {
            build.define("B128", None);
        }
    }

    for flag in PERF_FLAGS {
        build.flag(flag);
    }
    if wants_native_cpu_flags() {
        for flag in NATIVE_CPU_FLAGS {
            build.flag(flag);
        }
    }

    build.compile("lrslib");
}

fn wants_native_cpu_flags() -> bool {
    syn::wants_native_cpu_flags()
}

fn generate_bindings(layout: &LrsLayout) {
    let arith_dir = layout.source_dir.join("lrsarith-011");
    let use_gmp = env::var_os("CARGO_FEATURE_GMP").is_some();

    let header = "\
typedef __UINT8_TYPE__ uint8_t;\n\
typedef __UINT64_TYPE__ uint64_t;\n\
#include <stdint.h>\n\
#include <stdio.h>\n\
#include \"lrsrestart.h\"\n\
#include \"lrslib.h\"\n";

    let mut builder = bindgen::Builder::default()
        .header_contents("lrslib_rs.h", header)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Core lrslib lifecycle + enumeration.
        .allowlist_function("lrs_init")
        .allowlist_function("lrs_close")
        .allowlist_function("lrs_alloc_dat")
        .allowlist_function("lrs_free_dat")
        .allowlist_function("lrs_alloc_dic")
        .allowlist_function("lrs_free_dic")
        .allowlist_function("lrs_getfirstbasis")
        .allowlist_function("lrs_getnextbasis")
        .allowlist_function("lrs_getsolution")
        .allowlist_function("lrs_getvertex")
        .allowlist_function("lrs_getray")
        .allowlist_function("lrs_set_row")
        .allowlist_function("lrs_set_row_mp")
        // Minimal arithmetic helpers for setting/printing numbers.
        .allowlist_function("lrs_alloc_mp_vector")
        .allowlist_function("lrs_clear_mp_vector")
        .allowlist_function("lrs_alloc_mp_matrix")
        .allowlist_function("lrs_clear_mp_matrix")
        .allowlist_function("atomp")
        .allowlist_function("rattodouble")
        .allowlist_function("cpmp")
        .allowlist_function("cprat")
        // A couple of constants used when loading rows.
        .allowlist_var("GE")
        .allowlist_var("EQ")
        .allowlist_var("TRUE")
        .allowlist_var("FALSE")
        // Key lrslib data structures.
        .allowlist_type("lrs_dic")
        .allowlist_type("lrs_dat")
        .allowlist_type("lrs_restart_dat")
        .allowlist_type("lrs_mp")
        .allowlist_type("lrs_mp_t")
        .allowlist_type("lrs_mp_vector")
        .allowlist_type("lrs_mp_matrix")
        // Keep bindgen honest about our build-time configuration.
        .clang_arg("-x")
        .clang_arg("c")
        .clang_arg("-std=gnu99")
        .clang_arg(format!("--target={}", syn::target_triple()))
        .clang_arg(format!("-I{}", layout.source_dir.display()))
        .clang_arg(format!("-I{}", arith_dir.display()))
        .clang_arg("-DLRS_QUIET")
        .clang_arg("-DSIGNALS");

    for arg in syn::clang_system_include_args() {
        builder = builder.clang_arg(arg);
    }
    for arg in syn::clang_macos_sysroot_args() {
        builder = builder.clang_arg(arg);
    }

    let builder = if use_gmp {
        let mut builder = builder.clang_arg("-DGMP");
        if let Some(include_dir) = gmp_include_dir() {
            builder = builder.clang_arg(format!("-I{}", include_dir.display()));
        }
        builder
    } else {
        let builder = builder.clang_arg("-DLRSLONG").clang_arg("-DSAFE");
        if env::var("CARGO_CFG_TARGET_POINTER_WIDTH").as_deref() == Ok("64") {
            builder.clang_arg("-DB128")
        } else {
            builder
        }
    };

    let bindings = builder
        .generate()
        .expect("Unable to generate lrslib bindings with bindgen");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be provided"));
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn cache_root() -> PathBuf {
    syn::cache_root()
}

fn gmp_include_dir() -> Option<PathBuf> {
    let include = PathBuf::from(env::var_os("DEP_GMP_INCLUDE_DIR")?);
    include.join("gmp.h").is_file().then_some(include)
}
