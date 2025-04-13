from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_pdf):
    reader = PdfReader(input_pdf)
    num_pages = len(reader.pages)
    
    n = int(input("How many splits do you want? "))
    
    for i in range(n):
        page_range = input(f"Enter page range for split {i+1} (e.g., 5,10): ")
        pdf_name = input(f"Enter the name for split {i+1} (without .pdf extension): ")
        start, end = map(int, page_range.split(","))
        
        if start < 1 or end > num_pages or start > end:
            print("Invalid page range. Skipping...")
            continue
        
        writer = PdfWriter()
        for page in range(start-1, end):  # Pages are 0-indexed in PyPDF2
            writer.add_page(reader.pages[page])
        
        output_pdf = f"{pdf_name}.pdf"
        with open(output_pdf, "wb") as out_file:
            writer.write(out_file)
        
        print(f"Created: {output_pdf}")

if __name__ == "__main__":
    input_pdf = input("Enter the path to the PDF file: ")
    split_pdf(input_pdf)



