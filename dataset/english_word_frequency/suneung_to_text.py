import os
import glob
import io
import re
import pickle

import pdfminer.pdfinterp
import pdfminer.converter
import pdfminer.layout
import pdfminer.pdfpage

if not os.path.exists("./suneungs"):
    os.mkdir("./suneungs")

if not os.path.exists("./englsih_texts"):
    os.mkdir("./englsih_texts")

re_except_par = re.compile(r"\([a-zA-Z0-9]\)")
re_onlyEnglish = re.compile(r"[^a-zA-Z \n]")


def save_words():
    ret = []
    suneungs = glob.glob("suneungs/*.pdf")

    for path in suneungs:
        sng = path
        sng_bname = os.path.basename(os.path.splitext(sng)[0])

        resmgr = pdfminer.pdfinterp.PDFResourceManager()
        codec = "utf-8"
        laparams = pdfminer.layout.LAParams()

        fw = io.StringIO()
        # fw = open(f"./englsih_texts/{sng_bname}.txt", 'w', encoding='utf-8')
        device = pdfminer.converter.TextConverter(resmgr, fw, laparams=laparams)

        fp = open(sng, 'rb')
        interpreter = pdfminer.pdfinterp.PDFPageInterpreter(resmgr, device)

        for page in pdfminer.pdfpage.PDFPage.get_pages(fp):
            interpreter.process_page(page)

        fp.close()
        device.close()
        text = fw.getvalue()
        fw.close()
        text = re_except_par.sub('', text)
        text = re_onlyEnglish.sub('', text).strip().lower()
        words = text.split()
        words = list(filter(lambda x: x.strip(), words))
        ret.extend(words)

    with open("englsih_texts/data.pkl", "wb") as f:
        pickle.dump(ret, f)


if __name__ == '__main__':
    save_words()
