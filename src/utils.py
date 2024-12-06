from fpdf import FPDF

def create_report(data, output_file='report.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    
    for line in data:
        pdf.cell(200, 10, text=line, ln=True)
        
    pdf.output(output_file)