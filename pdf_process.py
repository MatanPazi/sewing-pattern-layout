from pathlib import Path
import re
import sys
import pikepdf

PATTERNS_DIR = Path(__file__).resolve().parent / "patterns"

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
    """Extract every OCG layer into its own multi-page PDF."""
    if out_dir is None:
        out_dir = pdf_path.parent / "extracted_layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"Output dir: {out_dir}")

    with pikepdf.open(pdf_path) as pdf:
        try:
            ocgs = list(pdf.Root.OCProperties.OCGs)
            print(f"Document contains {len(ocgs)} OCG(s)")
        except (AttributeError, KeyError):
            print("  → No layers found, skipping.")
            return 0

        # Collect all unique layer keys that actually appear in the document
        # (we will process them one by one)
        all_ocg_keys = set()
        for page in pdf.pages:
            try:
                props = page.Resources.Properties
                for key in props.keys():
                    # only keep keys that point to an OCG
                    obj = props[key]
                    if hasattr(obj, "get_object"):
                        obj = obj.get_object()
                    if obj.get("/Type") == "/OCG":
                        all_ocg_keys.add(key)
            except Exception:
                pass

        if not all_ocg_keys:
            print("  → No usable layer keys found on any page.")
            return 0

        print(f"Found {len(all_ocg_keys)} unique layer key(s) used in content")

    total_saved = 0

    # Process one layer at a time
    for ocg_key in sorted(all_ocg_keys, key=str):
        print(f"\n  ===== Extracting layer: {ocg_key} =====")

        with pikepdf.open(pdf_path) as pdf:
            # We keep every page, but strip content that does not belong to this layer
            for page_idx, page in enumerate(pdf.pages):
                commands = []
                extract_commands = True          # start outside any OC group
                inside_target_layer = False

                for operands, operator in pikepdf.parse_content_stream(page):
                    op_str = str(operator)

                    is_oc_start = (
                        op_str == "BDC"
                        and len(operands) >= 2
                        and str(operands[0]) == "/OC"
                    )

                    if is_oc_start:
                        current_key = operands[1]
                        if current_key == ocg_key:
                            extract_commands = True
                            inside_target_layer = True
                        else:
                            extract_commands = False
                            inside_target_layer = False
                        # Always keep the BDC itself so the structure stays valid
                        commands.append([operands, operator])
                        continue

                    if op_str == "EMC":
                        # Always keep the EMC
                        commands.append([operands, operator])
                        extract_commands = True
                        inside_target_layer = False
                        continue

                    # Keep the operator only if we are currently inside the target layer
                    # or if we are outside any OC group (optional – see note below)
                    if extract_commands:
                        commands.append([operands, operator])

                # Write the filtered content stream back to the page
                page.Contents = pdf.make_stream(
                    pikepdf.unparse_content_stream(commands)
                )

            # Determine a nice filename
            # Try to get the human-readable name from the first page that has it
            layer_name = str(ocg_key)
            try:
                for page in pdf.pages:
                    try:
                        props = page.Resources.Properties
                        if ocg_key in props:
                            layer_name = _get_layer_name(page, ocg_key)
                            break
                    except Exception:
                        pass
            except Exception:
                pass

            safe = _sanitize_filename(layer_name)
            out_path = out_dir / f"{safe}.pdf"

            # Avoid overwriting if the same layer name appears in multiple source PDFs
            counter = 1
            while out_path.exists():
                out_path = out_dir / f"{safe}_{counter}.pdf"
                counter += 1

            pdf.save(out_path)
            print(f"  → Saved multi-page layer as {out_path.name}")
            total_saved += 1

    print(f"\nFinished {pdf_path.name} – saved {total_saved} multi-page layer file(s)")
    return total_saved


def process_all_patterns():
    """
    Walk every sub-folder of the patterns directory and process every PDF inside.
    """
    print(f"Looking for pattern folders under: {PATTERNS_DIR}")

    # Only immediate sub-directories (the individual pattern folders)
    pattern_folders = [d for d in PATTERNS_DIR.iterdir() if d.is_dir()]

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
    #     PATTERNS_DIR / "ultimate_costume_creator_a0_format" / "ultimate_costume_creator_a0_format.pdf"
    # )
