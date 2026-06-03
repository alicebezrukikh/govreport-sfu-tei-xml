import lxml.etree as ET
import csv

# ПУТИ
xml_path = r"C:\Users\vladimir\Desktop\TEI\govreport-sfu-tei-xml\tei_reports_with_tables_formation\1851_report\1851_with_ids.xml"
csv_path = r"C:\Users\vladimir\Desktop\TEI\govreport-sfu-tei-xml\tei_reports_with_tables_formation\1851_report\dates_table.csv"
output_path = r"C:\Users\vladimir\Desktop\TEI\govreport-sfu-tei-xml\tei_reports_with_tables_formation\1851_report\1851_final_injected.xml"

# 1. Грузим XML
parser = ET.XMLParser(remove_blank_text=False)
tree = ET.parse(xml_path, parser)
root = tree.getroot()
ns = {'tei': 'http://www.tei-c.org/ns/1.0', 'xml': 'http://www.w3.org/XML/1998/namespace'}

# 2. Читаем CSV и инжектим ref
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        d_id = row['id'].strip()
        d_ref = row['event_ref'].strip()
        
        if d_ref:
            # Ищем элемент с таким xml:id и ставим ему ref
            element = root.xpath(f"//*[@xml:id='{d_id}']", namespaces=ns)
            if element:
                element[0].set("ref", d_ref)

# 3. Сохраняем
tree.write(output_path, encoding="utf-8", xml_declaration=True)
print(f"Готово. Файл сохранен: {output_path}")