
from lxml import etree

# 解析字符串/文件
xml_data = "<root><a>test</a></root>"
root = etree.fromstring(xml_data)  # 或 etree.parse('file.xml')

# 高级XPath查询
print(root.xpath("//a/text()"))  # 输出: ['test']

# 生成XML
new_elem = etree.Element("new", attr="value")
new_elem.text = "内容"
root.append(new_elem)
print(etree.tostring(root, pretty_print=True).decode())







