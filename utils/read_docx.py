import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx(file_path):
    try:
        docx_zip = zipfile.ZipFile(file_path)
        document_xml = docx_zip.read('word/document.xml')
        tree = ET.fromstring(document_xml)
        
        # The namespace for Word processing XML
        word_ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        
        paragraphs = []
        for paragraph in tree.iter(f'{word_ns}p'):
            texts = [node.text for node in paragraph.iter(f'{word_ns}t') if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        
        return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error reading docx: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = r"C:\Users\qwe\Desktop\7.1.docx"
    
    text = read_docx(path)
    with open('docx_output.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully wrote docx content to docx_output.txt")
