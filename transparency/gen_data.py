from num2words import num2words

langs = ['en', 'ja', 'dk', 'fr', 'es', 'ar', 'de', 'fi', 'ru', 'th', 'vi']

for lang in langs:
    with open("./data/numbers_" + lang + ".txt", "w+", encoding='utf-8') as f:
        for i in range(0, 10001):
            print(lang)
            f.write(num2words(i, lang=lang) + "\n")