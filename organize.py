from pathlib import Path
import shutil

SRC = Path("Celeb_twins_aligned")
DST = Path("Celeb_twins_folds")

DRY_RUN = True  # set to False after checking output
MOVE_INSTEAD_OF_COPY = False

folds = [
    ["Kinsman_twins", "Collins_twins", "Heo_twins", "Madden_twins", "Tyson-Sparks_twins", "Bellini_twins", "Stevens_twins", "Kaczynski_twins"],
    ["Goss_twins", "Huh_twins", "Bell_twins", "Lundqvist_twins", "Ananthan_twins", "Carvalho_twins", "Spencer_twins", "Vogt_twins"],
    ["Ashmore_twins", "Barber_twins", "Truong_twins", "Powney_twins", "Baker_twins", "Chapman_twins", "Ramsey_twins", "Venegas_twins"],
    ["Luttrell_twins", "Kaulitz_twins", "Hennessy_twins", "Daniel_twins", "Lejonhjarta_twins", "Bordier-Futerman_twins", "Quin_twins", "Thompson_twins"],
    ["Malek_twins", "Castro_twins", "Mowry_twins", "Heder_twins", "Hassan_twins", "Fontana_twins", "Peters_twins", "Olowofela_twins"],
    ["London_twins", "Jo_twins", "Velez_twins", "Grant_twins", "Cockrell_twins", "Herbert_twins", "Pahde_twins", "Nehls_twins"],
    ["Kelly_twins", "Harrison_twins", "Woo_twins", "Bryan_twins", "Kallur_twins", "Howe_twins", "Origliasso_twins", "Griffin_twins"],
    ["Phelps_twins", "Lopez_twins", "Haqq_twins", "Murray_twins", "Lind_twins", "Kilbey_twins", "Merrell_twins", "Duffer_twins"],
    ["Sprouse_twins", "Ramirez_twins", "Smith_twins", "Boer_twins", "Soska_twins", "Lamoureux_twins", "McKnight_twins", "Dobre_twins"],
    ["Treadaway_twins", "McCourty_twins", "Paul_twins", "D_Ambrosio_twins", "Quann_twins", "Lynch_twins", "Macedo_twins", "Brown_twins"],
]

# Check fold sizes
for i, fold in enumerate(folds):
    print(f"fold_{i}: {len(fold)} pairs")
    if len(fold) != 8:
        print(f"WARNING: fold_{i} does not have 8 pairs")

# Check duplicates
all_pairs = [pair for fold in folds for pair in fold]
duplicates = sorted({p for p in all_pairs if all_pairs.count(p) > 1})

if duplicates:
    print("\nDUPLICATES FOUND:")
    for d in duplicates:
        print(d)
else:
    print("\nNo duplicates found.")

# Check missing folders
missing = []

for pair in all_pairs:
    if not (SRC / pair).exists():
        missing.append(pair)

if missing:
    print("\nMISSING FROM SOURCE:")
    for m in missing:
        print(m)
else:
    print("\nNo missing source folders.")

# Create folds
for i, fold in enumerate(folds):
    fold_dir = DST / f"fold_{i}"
    print(f"\nCREATE: {fold_dir}")

    if not DRY_RUN:
        fold_dir.mkdir(parents=True, exist_ok=True)

    for pair_name in fold:
        src_path = SRC / pair_name
        dst_path = fold_dir / pair_name

        if not src_path.exists():
            print(f"  MISSING: {src_path}")
            continue

        if dst_path.exists():
            print(f"  SKIP exists: {dst_path}")
            continue

        action = "MOVE" if MOVE_INSTEAD_OF_COPY else "COPY"
        print(f"  {action}: {src_path} -> {dst_path}")

        if not DRY_RUN:
            if MOVE_INSTEAD_OF_COPY:
                shutil.move(str(src_path), str(dst_path))
            else:
                shutil.copytree(src_path, dst_path)

print("\nDone.")
print("Run once with DRY_RUN = True. If everything looks right, set DRY_RUN = False and rerun.")