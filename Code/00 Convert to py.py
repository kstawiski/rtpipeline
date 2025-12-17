import os
import nbformat

def convert_ipynb_to_py(directory="."):
    for file in os.listdir(directory):
        if file.endswith(".ipynb"):
            notebook_path = os.path.join(directory, file)
            output_path = os.path.splitext(notebook_path)[0] + ".py"

            try:
                # Read in binary mode, then decode explicitly
                with open(notebook_path, "rb") as nb_file:
                    nb_content = nb_file.read().decode("utf-8", errors="replace")  # Replaces problematic characters
                
                # Parse notebook content
                notebook = nbformat.reads(nb_content, as_version=4)

                # Write only code cells to the .py file
                with open(output_path, "w", encoding="utf-8") as py_file:
                    for cell in notebook.cells:
                        if cell.cell_type == "code":
                            py_file.write(cell.source + "\n\n")

                print(f"Converted: {file} → {output_path}")

            except Exception as e:
                print(f"Error converting {file}: {e}")

# Run the function in the current directory
convert_ipynb_to_py()

