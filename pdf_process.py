from pathlib import Path
import re
import sys
import pikepdf

SCRIPT_DIR = Path(__file__).resolve().parent   # = patterns/


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name))
    name = name.strip('. ')
    return name or "unnamed_layer"


def _get_layer_name(page, ocg_key) -> str:
    try:
        props = page.Resources.Properties
        ocg = props[ocg_key]
        if hasattr(ocg, "get_object"):
            ocg = ocg.get_object()
        return str(ocg.get("/Name", ocg_key))
    except Exception:
        return str(ocg_key)


def extract_layers_from_pdf(pdf_path: Path, out_dir: Path | None = None):
    """Extract every OCG layer from one PDF into its own files."""
    if out_dir is None:
        out_dir = pdf_path.parent / "extracted_layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"Output dir: {out_dir}")

    with pikepdf.open(pdf_path) as pdf:
        try:
            ocgs = pdf.Root.OCProperties.OCGs
            print(f"Document contains {len(ocgs)} OCG(s)")
        except (AttributeError, KeyError):
            print("  → No layers found, skipping.")
            return 0
        page_count = len(pdf.pages)

    hidden_operators = {
        "S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n",
        "Do", "sh", "Tj", "TJ", "m", "l", "c", "v", "y", "h", "re"
    }

    total_saved = 0

    for page_idx in range(page_count):
        print(f"\n  ----- Page {page_idx} -----")
        extracted_groups = []
        end_reached = False
        layer_pass = 0

        while not end_reached:
            commands = []
            extract_commands = True
            extracted_one = False
            current_ocg_key = None

            with pikepdf.open(pdf_path) as pdf:
                page = pdf.pages[page_idx]
                for j in range(len(pdf.pages) - 1, -1, -1):
                    if j != page_idx:
                        del pdf.pages[j]

                for operands, operator in pikepdf.parse_content_stream(page):
                    op_str = str(operator)

                    is_oc_start = (
                        op_str == "BDC"
                        and len(operands) >= 2
                        and str(operands[0]) == "/OC"
                    )

                    if is_oc_start:
                        ocg_key = operands[1]
                        if ocg_key not in extracted_groups and not extracted_one:
                            extracted_groups.append(ocg_key)
                            extract_commands = True
                            extracted_one = True
                            current_ocg_key = ocg_key
                            print(f"    Pass {layer_pass}: extracting {ocg_key}")
                        else:
                            extract_commands = False

                    if op_str == "EMC":
                        extract_commands = True
                        continue

                    if extract_commands or (
                        not extract_commands and op_str not in hidden_operators
                    ):
                        commands.append([operands, operator])

                if not extracted_one:
                    end_reached = True
                    print(f"    No more new layers on page {page_idx}")
                else:
                    page.Contents = pdf.make_stream(
                        pikepdf.unparse_content_stream(commands)
                    )

                    layer_name = _get_layer_name(page, current_ocg_key)
                    safe = _sanitize_filename(layer_name)

                    out_path = out_dir / f"{safe}.pdf"
                    counter = 1
                    while out_path.exists():
                        out_path = out_dir / f"{safe}_{counter}.pdf"
                        counter += 1

                    pdf.save(out_path)
                    print(f"    → Saved '{layer_name}' as {out_path.name}")
                    layer_pass += 1
                    total_saved += 1

    print(f"\nFinished {pdf_path.name} – saved {total_saved} layer file(s)")
    return total_saved


def process_all_patterns():
    """
    Walk every sub-folder of the patterns directory and process every PDF inside.
    """
    print(f"Looking for pattern folders under: {SCRIPT_DIR}")

    # Only immediate sub-directories (the individual pattern folders)
    pattern_folders = [d for d in SCRIPT_DIR.iterdir() if d.is_dir()]

    if not pattern_folders:
        print("No pattern folders found.")
        return

    for folder in sorted(pattern_folders):
        pdfs = list(folder.glob("*.pdf"))
        if not pdfs:
            print(f"\nSkipping {folder.name} (no PDF files)")
            continue

        for pdf in pdfs:
            # Skip anything already inside an extracted_layers folder
            if "extracted_layers" in pdf.parts:
                continue
            extract_layers_from_pdf(pdf)


if __name__ == "__main__":
    # -------------------------------------------------
    # Choose one of the two modes:
    # -------------------------------------------------

    # 1. Process EVERY pattern folder under patterns/
    process_all_patterns()

    # 2. Or process only one specific PDF (uncomment if you prefer)
    # extract_layers_from_pdf(
    #     SCRIPT_DIR / "ultimate_costume_creator_a0_format" / "ultimate_costume_creator_a0_format.pdf"
    # )