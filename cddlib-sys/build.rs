use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

use syntheon as syn;

const CDDLIB_TAG: &str = "0.94n";
const BACKEND_FEATURES: &[&str] = &[
    "CARGO_FEATURE_F64",
    "CARGO_FEATURE_GMP",
    "CARGO_FEATURE_GMPRATIONAL",
];
const PERF_FLAGS: &[&str] = &["-O3", "-DNDEBUG", "-g0", "-fomit-frame-pointer"];
const NATIVE_CPU_FLAGS: &[&str] = &["-march=native", "-mtune=native"];

fn vendor_dir() -> PathBuf {
    syn::vendor_dir()
}

#[derive(Clone)]
struct CddLayout {
    archive_path: PathBuf,
    source_dir: PathBuf,
    build_dir: PathBuf,
    install_dir: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    F64,
    GmpFloat,
    GmpRational,
}

impl Backend {
    fn cache_component(self) -> &'static str {
        match self {
            Backend::F64 => "f64",
            Backend::GmpFloat => "gmpfloat",
            Backend::GmpRational => "gmprational",
        }
    }

    fn lib_flavor(self) -> LibFlavor {
        match self {
            Backend::F64 => LibFlavor::F64,
            Backend::GmpFloat | Backend::GmpRational => LibFlavor::Gmp,
        }
    }
}

fn enabled_backends() -> Vec<Backend> {
    let mut backends = Vec::new();
    if env::var("CARGO_FEATURE_F64").is_ok() {
        backends.push(Backend::F64);
    }
    if env::var("CARGO_FEATURE_GMP").is_ok() {
        backends.push(Backend::GmpFloat);
    }
    if env::var("CARGO_FEATURE_GMPRATIONAL").is_ok() {
        backends.push(Backend::GmpRational);
    }
    backends
}

fn tools_backend(backends: &[Backend]) -> Option<Backend> {
    if env::var("CARGO_FEATURE_TOOLS").is_err() {
        return None;
    }
    if backends.contains(&Backend::GmpRational) {
        return Some(Backend::GmpRational);
    }
    if backends.contains(&Backend::GmpFloat) {
        return Some(Backend::GmpFloat);
    }
    if backends.contains(&Backend::F64) {
        return Some(Backend::F64);
    }
    None
}

fn main() {
    for feature in BACKEND_FEATURES {
        println!("cargo:rerun-if-env-changed={feature}");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TOOLS");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PIC");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-changed=build.rs");

    let backends = enabled_backends();
    if backends.is_empty() {
        panic!(
            "cddlib-sys: no numeric backend enabled; enable at least one of: f64, gmp, gmprational"
        );
    }
    let tools_backend = tools_backend(&backends);

    let header = "\
typedef __UINT8_TYPE__ uint8_t;\n\
typedef __UINT64_TYPE__ uint64_t;\n\
#include <stdint.h>\n\
#include <cddlib/setoper.h>\n\
#include <cddlib/cdd.h>\n";
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be provided by cargo"));

    println!("cargo:rustc-link-search=native={}", out_dir.display());

    if env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-lib=m");
    }

    for backend in backends {
        let layout = cdd_layout(backend);
        println!("cargo:rerun-if-changed={}", layout.archive_path.display());

        let install_dir = ensure_cddlib(&layout, backend);
        if tools_backend == Some(backend) {
            build_tools(&layout, backend, &install_dir);
        }

        let input_lib = cddlib_archive_path(&install_dir, backend);
        let prefix = backend_symbol_prefix(backend);
        let symbols = Rc::new(defined_symbols(&input_lib));

        let include_root = install_dir.join("include");
        let mut builder = bindgen::Builder::default()
            .header_contents("cddlib_rs.h", header)
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .parse_callbacks(Box::new(CddLinkNameCallbacks::new(prefix, symbols.clone())))
            .allowlist_function("dd_.*")
            .allowlist_type("dd_.*")
            .allowlist_var("dd_.*")
            .allowlist_function("set_.*")
            .allowlist_type("set_.*")
            .allowlist_var("set_.*")
            .blocklist_function("dd_Remove")
            .allowlist_function("mpq_.*")
            .allowlist_function("mpf_.*")
            .allowlist_function("__gmp.*")
            .allowlist_var("GMPRATIONAL");

        if backend.lib_flavor() == LibFlavor::Gmp {
            if let Some((gmp_include_dir, _)) = gmp_paths() {
                builder = builder.clang_arg(format!("-I{}", gmp_include_dir.display()));
            }
        }

        builder = match backend {
            Backend::F64 => builder
                .allowlist_function("ddd_.*")
                .blocklist_function("ddd_mpq_set_si"),
            Backend::GmpFloat => builder,
            Backend::GmpRational => builder.allowlist_function("ddd_mpq_set_si"),
        };

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
        builder = builder.clang_arg(format!("-I{}", include_root.display()));
        match backend {
            Backend::F64 => {}
            Backend::GmpFloat => {
                builder = builder.clang_arg("-DGMPFLOAT");
            }
            Backend::GmpRational => {
                builder = builder.clang_arg("-DGMPRATIONAL");
            }
        }

        let bindings = builder
            .generate()
            .expect("Unable to generate cddlib bindings with bindgen");

        let bindings_path = out_dir.join(bindings_filename(backend));
        fs::write(&bindings_path, bindings.to_string())
            .unwrap_or_else(|e| panic!("Couldn't write bindings {}: {e}", bindings_path.display()));

        let output_lib = out_dir.join(format!("lib{}.a", backend_lib_name(backend)));
        prefix_archive_symbols(&input_lib, &output_lib, prefix, symbols.as_ref());
        println!("cargo:rustc-link-lib=static={}", backend_lib_name(backend));
    }
}

#[derive(Debug)]
struct CddLinkNameCallbacks {
    prefix: &'static str,
    symbols: Rc<BTreeSet<String>>,
}

impl CddLinkNameCallbacks {
    fn new(prefix: &'static str, symbols: Rc<BTreeSet<String>>) -> Self {
        Self { prefix, symbols }
    }
}

impl bindgen::callbacks::ParseCallbacks for CddLinkNameCallbacks {
    fn generated_link_name_override(
        &self,
        item_info: bindgen::callbacks::ItemInfo<'_>,
    ) -> Option<String> {
        if !is_cddlib_symbol(item_info.name) || !self.symbols.contains(item_info.name) {
            return None;
        }
        Some(format!("{}{}", self.prefix, item_info.name))
    }
}

fn build_tools(layout: &CddLayout, backend: Backend, install_dir: &Path) {
    let src_dir = layout.source_dir.join("src");
    let bin_dir = install_dir.join("bin");
    fs::create_dir_all(&bin_dir).expect("failed to create cddlib tools directory");
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let include_dir = install_dir.join("include");
    let cdd_lib_dir = cddlib_lib_dir(install_dir, backend);
    let mut base_args = vec![
        "-O2".to_string(),
        "-std=c99".to_string(),
        format!("-I{}", include_dir.display()),
        format!("-L{}", cdd_lib_dir.display()),
    ];
    if backend.lib_flavor() == LibFlavor::Gmp {
        if let Some((gmp_include_dir, gmp_lib_dir)) = gmp_paths() {
            base_args.push(format!("-I{}", gmp_include_dir.display()));
            base_args.push(format!("-L{}", gmp_lib_dir.display()));
        }
    }
    let libs: Vec<String> = match backend {
        Backend::F64 => vec!["-lcdd".to_string()],
        Backend::GmpFloat | Backend::GmpRational => {
            vec!["-lcddgmp".to_string(), "-lgmp".to_string()]
        }
    };
    let tools = [
        "cddexec",
        "redcheck",
        "redexter",
        "redundancies",
        "redundancies_clarkson",
        "adjacency",
        "allfaces",
        "fourier",
        "lcdd",
        "projection",
        "scdd",
        "testcdd1",
        "testcdd2",
        "testlp1",
        "testlp2",
        "testlp3",
        "testshoot",
    ];
    for tool in tools {
        let src = src_dir.join(format!("{tool}.c"));
        if !src.exists() {
            continue;
        }
        let out = bin_dir.join(tool);
        let mut cmd = Command::new(&compiler);
        cmd.args(&base_args);
        cmd.arg(src);
        cmd.args(&libs);
        cmd.arg("-lm");
        cmd.arg("-o");
        cmd.arg(&out);
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to compile {tool}: {e}"));
        if !status.success() {
            panic!("{tool} build failed with status {status}");
        }
    }
    println!("cargo:rustc-env=CDDLIB_TOOLS_DIR={}", bin_dir.display());
}

fn bindings_filename(backend: Backend) -> &'static str {
    match backend {
        Backend::F64 => "bindings_f64.rs",
        Backend::GmpFloat => "bindings_gmpfloat.rs",
        Backend::GmpRational => "bindings_gmprational.rs",
    }
}

fn backend_lib_name(backend: Backend) -> &'static str {
    match backend {
        Backend::F64 => "cdd_f64",
        Backend::GmpFloat => "cdd_gmpfloat",
        Backend::GmpRational => "cdd_gmprational",
    }
}

fn backend_symbol_prefix(backend: Backend) -> &'static str {
    match backend {
        Backend::F64 => "cdd_f64_",
        Backend::GmpFloat => "cdd_gmpfloat_",
        Backend::GmpRational => "cdd_gmprational_",
    }
}

fn cddlib_lib_dir(install_dir: &Path, backend: Backend) -> PathBuf {
    let filename = match backend {
        Backend::F64 => "libcdd.a",
        Backend::GmpFloat | Backend::GmpRational => "libcddgmp.a",
    };
    for dir in ["lib", "lib64"] {
        let candidate = install_dir.join(dir);
        if candidate.join(filename).exists() {
            return candidate;
        }
    }
    panic!("missing {filename} under {}", install_dir.display());
}

fn cddlib_archive_path(install_dir: &Path, backend: Backend) -> PathBuf {
    let lib_dir = cddlib_lib_dir(install_dir, backend);
    match backend {
        Backend::F64 => lib_dir.join("libcdd.a"),
        Backend::GmpFloat | Backend::GmpRational => lib_dir.join("libcddgmp.a"),
    }
}

fn prefix_archive_symbols(input: &Path, output: &Path, prefix: &str, symbols: &BTreeSet<String>) {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("failed to create output directory for prefixed cddlib");
    }

    let redefine_path = output.with_extension("redefine.txt");
    let mut file = File::create(&redefine_path).unwrap_or_else(|e| {
        panic!(
            "failed to create symbol redefine file {}: {e}",
            redefine_path.display()
        )
    });
    for symbol in symbols {
        writeln!(file, "{symbol} {prefix}{symbol}")
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", redefine_path.display()));
    }

    let status = Command::new(objcopy_tool())
        .arg(format!("--redefine-syms={}", redefine_path.display()))
        .arg(input)
        .arg(output)
        .status()
        .unwrap_or_else(|e| panic!("failed to run objcopy: {e}"));
    if !status.success() {
        panic!("objcopy failed with status {status}");
    }

    let status = Command::new("ranlib")
        .arg(output)
        .status()
        .unwrap_or_else(|e| panic!("failed to run ranlib: {e}"));
    if !status.success() {
        panic!("ranlib failed with status {status}");
    }

    fs::remove_file(&redefine_path)
        .unwrap_or_else(|e| panic!("failed to remove {}: {e}", redefine_path.display()));
}

fn defined_symbols(archive: &Path) -> BTreeSet<String> {
    let output = Command::new(nm_tool())
        .arg("-g")
        .arg("--defined-only")
        .arg(archive)
        .output()
        .unwrap_or_else(|e| panic!("failed to run nm on {}: {e}", archive.display()));
    if !output.status.success() {
        panic!(
            "nm failed on {} with status {}",
            archive.display(),
            output.status
        );
    }
    let stdout = String::from_utf8(output.stdout).expect("nm output contained non-UTF8 bytes");

    let mut symbols = BTreeSet::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() || line.ends_with(':') {
            continue;
        }
        let Some(symbol) = line.split_whitespace().last() else {
            continue;
        };
        symbols.insert(symbol.to_string());
    }

    symbols
}

fn objcopy_tool() -> &'static str {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        require_tool("llvm-objcopy");
        return "llvm-objcopy";
    }
    "objcopy"
}

fn nm_tool() -> &'static str {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        require_tool("llvm-nm");
        return "llvm-nm";
    }
    "nm"
}

fn require_tool(tool: &str) {
    if Command::new(tool).arg("--version").status().is_ok() {
        return;
    }
    panic!(
        "{tool} is required for cddlib-sys on this platform (install LLVM, e.g. `brew install llvm`)."
    );
}

fn is_cddlib_symbol(name: &str) -> bool {
    name.starts_with("dd_") || name.starts_with("set_") || name.starts_with("ddd_")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LibFlavor {
    F64,
    Gmp,
}

fn ensure_cddlib(layout: &CddLayout, backend: Backend) -> PathBuf {
    if has_backend_lib(&layout.install_dir, backend) {
        return layout.install_dir.clone();
    }
    ensure_cdd_source(layout);
    if backend == Backend::GmpFloat {
        patch_gmpfloat_sources(&layout.source_dir);
    }
    build_cddlib(layout, backend)
}

fn ensure_cdd_source(layout: &CddLayout) -> PathBuf {
    if layout.source_dir.join("configure").exists() {
        return layout.source_dir.clone();
    }
    if let Some(parent) = layout.source_dir.parent() {
        fs::create_dir_all(parent).expect("failed to create cddlib source parent directory");
    }
    let root = layout
        .source_dir
        .parent()
        .unwrap_or_else(|| panic!("missing parent for {}", layout.source_dir.display()));
    extract_archive(&layout.archive_path, root);
    if layout.source_dir.join("configure").exists() {
        return layout.source_dir.clone();
    }
    panic!(
        "cddlib source tree not found under {} after extraction",
        layout.source_dir.display()
    );
}

fn patch_gmpfloat_sources(root: &Path) {
    let lib_src = root.join("lib-src");
    let replacements = [
        (
            lib_src.join("cddmp.c"),
            "mpf_set_si(dd_minusone,-1L,1U);",
            "mpf_set_si(dd_minusone,-1L);",
        ),
        (
            lib_src.join("cddmp_f.c"),
            "mpf_set_si(ddf_minusone,-1L,1U);",
            "mpf_set_si(ddf_minusone,-1L);",
        ),
        (
            lib_src.join("cddmp.h"),
            "#define dd_set_si2(a, b, c)     mpf_set_si(a,b,c)    /* gmp 3.1 or higher */",
            "#define dd_set_si2(a, b, c)     do { mpf_set_si(a,b); mpf_div_ui(a,a,c); } while (0)",
        ),
        (
            lib_src.join("cddmp_f.h"),
            "#define ddf_set_si2(a, b, c)     mpf_set_si(a,b,c)    /* gmp 3.1 or higher */",
            "#define ddf_set_si2(a, b, c)     do { mpf_set_si(a,b); mpf_div_ui(a,a,c); } while (0)",
        ),
    ];
    for (path, needle, replacement) in replacements {
        if path.exists() {
            replace_once(&path, needle, replacement);
        }
    }
}

fn build_cddlib(layout: &CddLayout, backend: Backend) -> PathBuf {
    if has_backend_lib(&layout.install_dir, backend) {
        return layout.install_dir.clone();
    }

    fs::create_dir_all(&layout.build_dir).expect("failed to create cddlib build directory");

    let mut configure = Command::new(layout.source_dir.join("configure"));
    let needs_pic = env::var_os("CARGO_FEATURE_PIC").is_some();
    let perf_flags = if needs_pic {
        format!("{} -fPIC", perf_flag_string())
    } else {
        perf_flag_string()
    };
    configure
        .arg(format!("--prefix={}", layout.install_dir.display()))
        .arg("--enable-shared=no")
        .arg("--enable-static=yes")
        .args(needs_pic.then_some("--with-pic"))
        .current_dir(&layout.build_dir)
        .env("CFLAGS", &perf_flags)
        .env("CXXFLAGS", &perf_flags)
        .env("ac_cv_path_lt_DD", "/bin/dd")
        .env("lt_cv_truncate_bin", "sed -e 4q");

    if backend.lib_flavor() == LibFlavor::Gmp {
        if let Some((gmp_include_dir, gmp_lib_dir)) = gmp_paths() {
            let cppflags = format!("-I{}", gmp_include_dir.display());
            let ldflags = format!("-L{}", gmp_lib_dir.display());
            configure.env("CPPFLAGS", cppflags).env("LDFLAGS", ldflags);
        }
    }
    run(&mut configure, "cddlib configure failed");

    if backend == Backend::GmpFloat {
        patch_gmpfloat_sources(&layout.build_dir);
        rewrite_gmp_makefiles(&layout.build_dir);
    }

    let jobs = parallel_jobs();
    let mut make = Command::new("make");
    apply_parallel(&mut make, jobs);
    make.current_dir(&layout.build_dir);
    run(&mut make, "cddlib make failed");

    let mut make_install = Command::new("make");
    make_install
        .arg("install")
        .current_dir(&layout.build_dir)
        .env("CMAKE_BUILD_PARALLEL_LEVEL", jobs.to_string());
    apply_parallel(&mut make_install, jobs);
    run(&mut make_install, "cddlib make install failed");

    if !has_backend_lib(&layout.install_dir, backend) {
        panic!(
            "cddlib build did not produce the requested backend ({:?}) under {}",
            backend,
            layout.install_dir.display()
        );
    }

    layout.install_dir.clone()
}

fn cdd_layout(backend: Backend) -> CddLayout {
    let archive_path = vendor_dir().join(format!("cddlib-{CDDLIB_TAG}.tar.gz"));
    if !archive_path.is_file() {
        panic!(
            "missing vendored cddlib archive at {}",
            archive_path.display()
        );
    }
    let cache_root = cache_root();
    let needs_pic = env::var_os("CARGO_FEATURE_PIC").is_some();
    let dir_key = format!(
        "{}-{}{}",
        syn::sanitize_component(CDDLIB_TAG),
        backend.cache_component(),
        if needs_pic { "-pic" } else { "" },
    );
    let root = cache_root.join(dir_key);
    CddLayout {
        archive_path,
        source_dir: root.join(format!("cddlib-{CDDLIB_TAG}")),
        build_dir: root.join("build"),
        install_dir: root.join("install"),
    }
}

fn cache_root() -> PathBuf {
    syn::cache_root()
}

fn available_lib_dirs(root: &Path) -> Vec<(LibFlavor, PathBuf)> {
    let mut dirs = Vec::new();
    for dir in ["lib", "lib64"] {
        let path = root.join(dir);
        if path.join("libcdd.a").exists() {
            dirs.push((LibFlavor::F64, path.clone()));
        }
        if path.join("libcddgmp.a").exists() {
            dirs.push((LibFlavor::Gmp, path.clone()));
        }
    }
    dirs
}

fn has_backend_lib(root: &Path, backend: Backend) -> bool {
    let flavor = backend.lib_flavor();
    available_lib_dirs(root)
        .into_iter()
        .any(|(found, _)| found == flavor)
}

fn replace_once(path: &Path, needle: &str, replacement: &str) {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    if contents.contains(needle) {
        let updated = contents.replace(needle, replacement);
        fs::write(path, updated)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
    } else if !contents.contains(replacement) {
        panic!(
            "expected to replace {needle} in {}, but it was not found",
            path.display()
        );
    }
}

fn rewrite_gmp_makefiles(build_dir: &Path) {
    let lib_src_makefile = build_dir.join("lib-src/Makefile");
    rewrite_makefile_flag(&lib_src_makefile, "-DGMPRATIONAL", "-DGMPFLOAT");

    let src_makefile = build_dir.join("src/Makefile");
    rewrite_makefile_flag(&src_makefile, "-DGMPRATIONAL", "-DGMPFLOAT");
}

fn rewrite_makefile_flag(path: &Path, needle: &str, replacement: &str) {
    replace_once(path, needle, replacement);
}

fn perf_flag_string() -> String {
    perf_flags().collect::<Vec<_>>().join(" ")
}

fn perf_flags() -> impl Iterator<Item = &'static str> {
    PERF_FLAGS
        .iter()
        .copied()
        .chain(native_cpu_flags().iter().copied())
}

fn native_cpu_flags() -> &'static [&'static str] {
    if wants_native_cpu_flags() {
        NATIVE_CPU_FLAGS
    } else {
        &[]
    }
}

fn wants_native_cpu_flags() -> bool {
    syn::wants_native_cpu_flags()
}

fn parallel_jobs() -> usize {
    syn::parallel_jobs()
}

fn apply_parallel(cmd: &mut Command, jobs: usize) {
    syn::apply_parallel(cmd, jobs);
}

fn extract_archive(archive_path: &Path, out_dir: &Path) {
    syn::extract_tar_gz(archive_path, out_dir);
}

fn run(cmd: &mut Command, err: &str) {
    syn::run(cmd, err);
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
