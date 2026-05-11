from pathlib import Path
import numpy as np

allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")

def afsample2_a3m(a3m_file, seed=0, frac=0.1, rowColumnSwitch="column"):
    rng = np.random.default_rng(seed)
    a3m_file = Path(a3m_file)

    lines = a3m_file.read_text().splitlines()
    records = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">"):
            records.append([lines[i], lines[i + 1].strip()])
        else:
            console.print(f"[red bold] Expected a header line starting with \">\" at position {i}. Instead line was {lines[i]}[/]")
            raise RuntimeError(f"Failed to create {rowColumnSwitch}-masked a3m file")
    
    # About this replacement logic, ChatGPT said that lower-case letter can appear in the fasta sequences of the MSA
    # eas xtra characters, which do not count towards the sequence length, which would cause the lines to have unequal
    # length. At a glance I didn't see this in our files, so I would just try this more restrictive logic for now.
    if rowColumnSwitch == "column":
        n_cols = len(records[0][1])
        rand_cols = set(rng.choice(n_cols, int(n_cols * frac), replace=False))

        for i in range(1, len(records)):
            seq = list(records[i][1])
            if len(seq) != len(records[0][1]):
                console.print(f"[red bold] Aligned sequence length ({len(seq)}) does not match input sequence length ({len(records[0][1])})[/]")
                raise RuntimeError("Failed to create row-masked a3m file")
            for j, char in enumerate(seq):
                if (char not in allowed):
                    console.print(f"[red bold] Unexpected character in aligned fasta sequence {i} at position {j}: {char}[/]")
                    raise RuntimeError("Failed to create column-masked a3m file")
                if j in rand_cols:
                    seq[j] = "X"
            records[i][1] = "".join(seq)

    elif rowColumnSwitch == "row":
        n_rows = len(records)
        rand_rows = set(rng.choice(n_rows - 1, int((n_rows - 1) * frac), replace=False))

        for i in range(1, n_rows):
            seq = list(records[i][1])
            if len(seq) != len(records[0][1]):
                console.print(f"[red bold] Aligned sequence length ({len(seq)}) does not match input sequence length ({len(records[0][1])})[/]")
                raise RuntimeError("Failed to create row-masked a3m file")
            for j, char in enumerate(seq):
                if (char not in allowed):
                    console.print(f"[red bold] Unexpected character in aligned fasta sequence {i} at position {j}: {char}[/]")
                    raise RuntimeError("Failed to create row-masked a3m file")
            if i in rand_rows:
                records[i][1] = "X" * len(seq)
    
    else:
        raise ValueError("rowColumnSwitch should be \"column\" or \"row\".")

    outFileName = a3m_file.stem + rowColumnSwitch + "Masked" + str(seed) + ".a3m"
    out = a3m_file.with_name(outFileName)
    out.write_text("\n".join(f"{h}\n{s}" for h, s in records) + "\n")
    console.print(f"[green]Wrote {rowColumnSwitch}-masked file {outFileName}[/]")
    return out