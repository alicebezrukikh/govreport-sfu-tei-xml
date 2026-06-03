import lxml.etree as ET
import csv
import os

def export_dates_to_csv(input_xml, output_csv, output_xml_with_ids):
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    parser = ET.XMLParser(remove_blank_text=False)
    tree = ET.parse(input_xml, parser)
    root = tree.getroot()

    dates = root.xpath("//tei:date", namespaces=namespaces)
    
    csv_data = []

    for i, date_tag in enumerate(dates):
        # 1. Генерируем уникальный ID для каждой даты
        date_id = f"d{i+1}"
        date_tag.set("{http://www.w3.org/XML/1998/namespace}id", date_id)
        
        # 2. Собираем данные
        date_text = "".join(date_tag.itertext()).strip()
        
        # Берем контекст (родительский абзац)
        parent = date_tag.getparent()
        full_text = " ".join("".join(parent.itertext()).split()) # очистка пробелов
        
        # Обрезаем контекст для CSV, чтобы он не был гигантским (по 40 слов вокруг)
        # Но для точности оставим побольше
        csv_data.append({
            'id': date_id,
            'date_text': date_text,
            'event_ref': '',  # Сюда ты будешь писать значения
            'context': full_text
        })

    # Сохраняем CSV (разделитель - точка с запятой)
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'date_text', 'event_ref', 'context'], delimiter=';')
        writer.writeheader()
        writer.writerows(csv_data)

    # Сохраняем XML с проставленными ID
    tree.write(output_xml_with_ids, encoding="utf-8", xml_declaration=True)
    
    print(f"Готово!")
    print(f"1. CSV для редактирования: {output_csv}")
    print(f"2. XML с ID (рабочий): {output_xml_with_ids}")

# ПУТИ
path = r"C:\Users\vladimir\Desktop\TEI\govreport-sfu-tei-xml\tei_reports_with_tables_formation\1851_report"
input_f = os.path.join(path, "1851_final_marked.xml")
out_csv = os.path.join(path, "dates_table.csv")
out_xml = os.path.join(path, "1851_with_ids.xml")

export_dates_to_csv(input_f, out_csv, out_xml)