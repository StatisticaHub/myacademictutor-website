from pathlib import Path

lesson_path = Path("public/learning-hub/machine-learning-biostatistics/module-1/1-1-what-is-machine-learning-in-biostatistics.html")
css_path = Path("public/learning-hub/machine-learning-biostatistics/assets/course.css")
note_path = Path("note1-section.html")
extra_css_path = Path("note1-extra-css.css")

html = lesson_path.read_text(encoding="utf-8")
note = note_path.read_text(encoding="utf-8")

start = html.find('<section class="lesson-section" id="notes">')
if start != -1:
    end = html.find('<section class="section">', start)
    if end != -1:
        html = html[:start] + html[end:]

marker = '<section class="section">\n            <div class="next-lesson-card">'
if marker not in html:
    raise SystemExit("Could not find next lesson card marker. Insert note1-section.html manually before the next lesson card.")

html = html.replace(marker, note + "\n          " + marker, 1)
lesson_path.write_text(html, encoding="utf-8")

css = css_path.read_text(encoding="utf-8")
extra = extra_css_path.read_text(encoding="utf-8")
if "/* Notes and table enhancements */" not in css:
    css_path.write_text(css + "\n" + extra, encoding="utf-8")

print("Updated Lesson 1 with Note 1 section and CSS.")
