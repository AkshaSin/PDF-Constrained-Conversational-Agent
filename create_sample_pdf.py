"""
Script to generate the sample evaluation PDF for the PDF-Constrained Agent.
Run with: python create_sample_pdf.py
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

OUTPUT_PATH = "sample.pdf"

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    rightMargin=2*cm,
    leftMargin=2*cm,
    topMargin=2*cm,
    bottomMargin=2*cm
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "CustomTitle", parent=styles["Title"],
    fontSize=22, spaceAfter=12, textColor=colors.HexColor("#1a1a2e")
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=16, spaceAfter=8, textColor=colors.HexColor("#16213e")
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=13, spaceAfter=6, textColor=colors.HexColor("#0f3460")
)
body_style = ParagraphStyle(
    "Body", parent=styles["BodyText"],
    fontSize=11, leading=16, spaceAfter=8, alignment=TA_JUSTIFY
)
caption_style = ParagraphStyle(
    "Caption", parent=styles["Normal"],
    fontSize=9, textColor=colors.grey, spaceAfter=12, alignment=TA_CENTER
)

story = []

# ─── PAGE 1: Cover & Executive Summary ─────────────────────────────────────
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph("Global Climate Change Report 2024", title_style))
story.append(Paragraph("An Analysis of Causes, Impacts, and Mitigation Strategies", styles["Heading2"]))
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph(
    "Published by: The International Climate Research Consortium (ICRC) | Edition: Fifth Annual Report",
    caption_style
))
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph("Executive Summary", h1_style))
story.append(Paragraph(
    "Global average surface temperatures have risen by approximately 1.1 degrees Celsius above "
    "pre-industrial levels as of 2024. This warming is unequivocally driven by human activities, "
    "primarily the combustion of fossil fuels such as coal, oil, and natural gas, which releases "
    "carbon dioxide (CO2) and other greenhouse gases into the atmosphere. The concentration of "
    "atmospheric CO2 has reached 421 parts per million (ppm), the highest level in over 800,000 years.",
    body_style
))
story.append(Paragraph(
    "The consequences are far-reaching: rising sea levels threatening coastal communities, "
    "increased frequency and intensity of extreme weather events, disruption of ecosystems, "
    "and threats to global food security. Without immediate and sustained action to reduce "
    "greenhouse gas emissions, global temperatures are projected to rise by 2.5 to 4.0 degrees "
    "Celsius by the end of the century, leading to catastrophic and largely irreversible impacts.",
    body_style
))
story.append(Paragraph(
    "This report recommends a tripartite response strategy: aggressive deployment of renewable "
    "energy sources to decarbonize the power sector; implementation of carbon pricing mechanisms "
    "to drive economy-wide emissions reductions; and substantial investment in adaptation "
    "measures to protect vulnerable communities from unavoidable climate impacts.",
    body_style
))
story.append(PageBreak())

# ─── PAGE 2: Causes of Climate Change ─────────────────────────────────────
story.append(Paragraph("Section 1: Primary Causes of Climate Change", h1_style))
story.append(Paragraph("1.1 The Greenhouse Effect and Human Emissions", h2_style))
story.append(Paragraph(
    "The greenhouse effect is a natural process where certain gases in the Earth's atmosphere "
    "trap heat from the sun, keeping the planet warm enough to support life. The primary "
    "greenhouse gases include carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), and "
    "fluorinated gases. Human industrial activities since the mid-18th century have dramatically "
    "amplified this natural effect, leading to the current climate crisis.",
    body_style
))
story.append(Paragraph(
    "Fossil fuel combustion for electricity and heat generation accounts for the largest share "
    "of global greenhouse gas emissions at 34%. Transportation contributes 16%, while industry "
    "accounts for 24%. Agriculture, forestry, and land use change collectively contribute 22%, "
    "with methane from livestock and nitrous oxide from fertilizers being the primary drivers "
    "in this category. Buildings account for the remaining 6% of global emissions.",
    body_style
))
story.append(Paragraph("1.2 Deforestation and Land Use Change", h2_style))
story.append(Paragraph(
    "Forests serve as critical carbon sinks, absorbing approximately 2.6 billion tonnes of CO2 "
    "per year. Deforestation, primarily for agricultural expansion and logging, not only destroys "
    "this carbon storage capacity but also releases stored carbon into the atmosphere. The Amazon "
    "rainforest, often called the 'lungs of the Earth,' has lost approximately 17% of its original "
    "extent due to deforestation, a figure that climate scientists warn is approaching a dangerous "
    "tipping point beyond which the forest could transition to a savanna ecosystem.",
    body_style
))
story.append(PageBreak())

# ─── PAGE 3: Observed Impacts ──────────────────────────────────────────────
story.append(Paragraph("Section 2: Observed and Projected Impacts", h1_style))
story.append(Paragraph("2.1 Rising Sea Levels", h2_style))
story.append(Paragraph(
    "Global mean sea levels have risen by approximately 21 centimeters since 1900, with the rate "
    "of rise accelerating significantly in recent decades. The current rate is approximately 3.7 "
    "millimeters per year. This rise is driven by two main factors: thermal expansion of warming "
    "ocean waters (accounting for 42% of observed rise) and the melting of land-based ice sheets "
    "and glaciers (accounting for 58%). Low-lying island nations such as Tuvalu, Kiribati, and "
    "the Maldives face existential threats from rising seas and increased storm surge.",
    body_style
))
story.append(Paragraph("2.2 Extreme Weather Events", h2_style))
story.append(Paragraph(
    "Scientific evidence increasingly links climate change to the intensification of extreme "
    "weather events. The frequency of Category 4 and 5 hurricanes has increased by 25-30% since "
    "the 1980s. Heatwaves that would have occurred once every 50 years in a pre-industrial climate "
    "now occur every 10 years on average. Precipitation extremes are also intensifying: wet regions "
    "are getting wetter and dry regions are getting drier, increasing both flood and drought risks.",
    body_style
))
story.append(Paragraph("2.3 Arctic and Antarctic Ice Loss", h2_style))
story.append(Paragraph(
    "The Arctic is warming at a rate approximately four times faster than the global average, a "
    "phenomenon known as Arctic Amplification. Arctic sea ice extent has declined by approximately "
    "13% per decade since satellite records began in 1979. The Greenland Ice Sheet is losing mass "
    "at an average rate of 280 billion tonnes per year, while the Antarctic Ice Sheet loses "
    "approximately 150 billion tonnes annually. If the West Antarctic Ice Sheet were to collapse "
    "entirely, it would raise global sea levels by approximately 3.3 meters.",
    body_style
))
story.append(PageBreak())

# ─── PAGE 4: Mitigation & Adaptation ──────────────────────────────────────
story.append(Paragraph("Section 3: Mitigation and Adaptation Strategies", h1_style))
story.append(Paragraph("3.1 Renewable Energy Transition", h2_style))
story.append(Paragraph(
    "The rapid deployment of renewable energy is the cornerstone of climate mitigation. Solar "
    "photovoltaic (PV) and wind energy are now the cheapest sources of new electricity generation "
    "in most of the world. The cost of solar PV has fallen by 89% over the past decade, and "
    "wind power costs have dropped by 70%. Global renewable energy capacity reached 3,372 gigawatts "
    "in 2023. To limit warming to 1.5 degrees Celsius, renewable energy must supply at least 85% "
    "of global electricity by 2050, up from the current 30%.",
    body_style
))
story.append(Paragraph("3.2 Carbon Pricing", h2_style))
story.append(Paragraph(
    "Carbon pricing is widely considered the most economically efficient policy instrument for "
    "reducing greenhouse gas emissions. It works by putting a direct price on carbon emissions, "
    "incentivizing businesses and individuals to reduce their carbon footprint. There are two main "
    "forms: a carbon tax, which sets a direct price on emissions, and an emissions trading system "
    "(ETS), also known as cap-and-trade, which sets a cap on total emissions and allows entities "
    "to buy and sell allowances. As of 2024, 73 carbon pricing initiatives are in place globally, "
    "covering approximately 23% of global greenhouse gas emissions.",
    body_style
))
story.append(Paragraph("3.3 The Role of Nature-Based Solutions", h2_style))
story.append(Paragraph(
    "Nature-based solutions (NbS) — such as reforestation, wetland restoration, and sustainable "
    "agricultural practices — can contribute up to 30% of the emissions reductions needed by 2030 "
    "to meet the Paris Agreement goals. Reforestation programs in particular are gaining momentum: "
    "the Bonn Challenge, a global effort to restore 350 million hectares of degraded land by 2030, "
    "has already seen pledges covering 210 million hectares. However, scientists caution that "
    "nature-based solutions are not a substitute for deep, rapid reductions in fossil fuel emissions.",
    body_style
))
story.append(PageBreak())

# ─── PAGE 5: Multilingual Section & Key Statistics ─────────────────────────
story.append(Paragraph("Section 4: Global Policy Context and Key Statistics", h1_style))
story.append(Paragraph("4.1 The Paris Agreement", h2_style))
story.append(Paragraph(
    "The Paris Agreement, adopted in December 2015 under the United Nations Framework Convention "
    "on Climate Change (UNFCCC), represents the landmark international accord on climate action. "
    "Its central aim is to limit global average temperature increases to well below 2 degrees "
    "Celsius above pre-industrial levels, and to pursue efforts to limit warming to 1.5 degrees "
    "Celsius. As of 2024, 195 countries have ratified the agreement. Each country submits "
    "Nationally Determined Contributions (NDCs) outlining its climate action plans. "
    "However, a 2023 analysis found that current NDCs are collectively insufficient and would "
    "lead to warming of approximately 2.5 to 2.9 degrees Celsius by 2100.",
    body_style
))

story.append(Paragraph("4.2 Key Statistics at a Glance", h2_style))
table_data = [
    ["Indicator", "Value", "Year"],
    ["Global Temperature Rise", "1.1°C above pre-industrial", "2024"],
    ["Atmospheric CO2 Concentration", "421 parts per million (ppm)", "2024"],
    ["Annual Sea Level Rise Rate", "3.7 mm per year", "2023"],
    ["Renewable Energy Global Share", "30% of electricity generation", "2023"],
    ["Annual Arctic Sea Ice Loss", "13% per decade since 1979", "2024"],
    ["Countries with Carbon Pricing", "73 initiatives globally", "2024"],
    ["Cost Reduction of Solar PV", "89% decline over last decade", "2023"],
]
table = Table(table_data, colWidths=[7*cm, 7*cm, 3*cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 10),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ("FONTSIZE", (0, 1), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(table)
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph("4.3 Multilingual Summary (Bonus: Non-English Grounding Test)", h2_style))
story.append(Paragraph(
    "<b>Hindi (हिंदी):</b> जलवायु परिवर्तन आज के समय की सबसे बड़ी वैश्विक चुनौती है। "
    "वायुमंडल में CO2 की सांद्रता 421 ppm तक पहुँच गई है, जो पिछले 800,000 वर्षों में "
    "सबसे अधिक है। इसे नियंत्रित करने के लिए नवीकरणीय ऊर्जा को अपनाना अनिवार्य है।",
    body_style
))
story.append(Paragraph(
    "<b>French (Français):</b> Le changement climatique est la plus grande menace environnementale "
    "de notre époque. La température moyenne mondiale a augmenté de 1,1 degré Celsius depuis "
    "l'ère préindustrielle. L'Accord de Paris vise à limiter ce réchauffement à 1,5 degré Celsius.",
    body_style
))

doc.build(story)
print(f"Sample PDF created successfully at: {OUTPUT_PATH}")
