"""
60 synthetic cargo manifest cases for the CBIC RL Environment.
Seed=42, deterministic. Do not modify.
Covers: 20 clean, 20 easy (1 anomaly), 12 medium (2 anomalies), 8 hard (3+ anomalies).
"""

from environment.models import (
    CargoManifest, CaseMetadata, CustomsCase,
    AnomalyType, Channel
)

CASES: list[CustomsCase] = [

    # =========================================================
    # CLEAN CASES (20) — no anomalies, correct channel = GREEN
    # =========================================================

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0001",
            port_of_entry="JNPT",
            importer_name="Tata Consumer Products Ltd",
            iec_code="IEC0588012345",
            iec_age_months=84,
            country_of_origin="USA",
            country_of_export="USA",
            commodity="Coffee Beans",
            hs_code="0901.11",
            declared_weight_kg=18000,
            declared_value_usd=54000,
            market_value_usd=55000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["USA"],
            container_type="20GP",
            description="Arabica coffee beans, roasting grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-001",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0002",
            port_of_entry="Mundra",
            importer_name="Reliance Industries Ltd",
            iec_code="IEC0312098765",
            iec_age_months=120,
            country_of_origin="Saudi Arabia",
            country_of_export="Saudi Arabia",
            commodity="Crude Oil",
            hs_code="2709.00",
            declared_weight_kg=500000,
            declared_value_usd=350000,
            market_value_usd=360000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Saudi Arabia"],
            container_type="TANK",
            description="Light sweet crude oil, API 38",
        ),
        metadata=CaseMetadata(
            case_id="CASE-002",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0003",
            port_of_entry="Chennai",
            importer_name="Infosys BPO Ltd",
            iec_code="IEC0719054321",
            iec_age_months=96,
            country_of_origin="Japan",
            country_of_export="Japan",
            commodity="Laptop Computers",
            hs_code="8471.30",
            declared_weight_kg=2400,
            declared_value_usd=480000,
            market_value_usd=490000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Japan"],
            container_type="20GP",
            description="Business laptops, 15.6 inch, Core i7",
        ),
        metadata=CaseMetadata(
            case_id="CASE-003",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0004",
            port_of_entry="Kolkata",
            importer_name="ITC Limited",
            iec_code="IEC0304087654",
            iec_age_months=144,
            country_of_origin="Brazil",
            country_of_export="Brazil",
            commodity="Raw Cotton",
            hs_code="5201.00",
            declared_weight_kg=40000,
            declared_value_usd=56000,
            market_value_usd=57000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Brazil"],
            container_type="40GP",
            description="Long staple cotton, Grade A",
        ),
        metadata=CaseMetadata(
            case_id="CASE-004",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0005",
            port_of_entry="JNPT",
            importer_name="Mahindra & Mahindra Ltd",
            iec_code="IEC0518023456",
            iec_age_months=108,
            country_of_origin="Germany",
            country_of_export="Germany",
            commodity="CNC Machine Tools",
            hs_code="8457.10",
            declared_weight_kg=15000,
            declared_value_usd=750000,
            market_value_usd=760000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Germany"],
            container_type="40HC",
            description="5-axis CNC machining centers",
        ),
        metadata=CaseMetadata(
            case_id="CASE-005",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0006",
            port_of_entry="Mundra",
            importer_name="Adani Ports Special Economic Zone",
            iec_code="IEC0602034567",
            iec_age_months=60,
            country_of_origin="China",
            country_of_export="China",
            commodity="Solar Panels",
            hs_code="8541.40",
            declared_weight_kg=25000,
            declared_value_usd=200000,
            market_value_usd=205000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China"],
            container_type="40HC",
            description="Monocrystalline solar modules, 400W",
        ),
        metadata=CaseMetadata(
            case_id="CASE-006",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0007",
            port_of_entry="Chennai",
            importer_name="Hyundai Motor India Ltd",
            iec_code="IEC0998045678",
            iec_age_months=132,
            country_of_origin="South Korea",
            country_of_export="South Korea",
            commodity="Auto Spare Parts",
            hs_code="8708.99",
            declared_weight_kg=8000,
            declared_value_usd=120000,
            market_value_usd=122000,
            previous_violations=0,
            related_party=True,
            related_party_disclosed=True,
            routing_countries=["South Korea"],
            container_type="20GP",
            description="OEM engine components, transmission parts",
        ),
        metadata=CaseMetadata(
            case_id="CASE-007",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0008",
            port_of_entry="JNPT",
            importer_name="Cipla Ltd",
            iec_code="IEC0886056789",
            iec_age_months=180,
            country_of_origin="Switzerland",
            country_of_export="Switzerland",
            commodity="Active Pharmaceutical Ingredients",
            hs_code="2941.10",
            declared_weight_kg=500,
            declared_value_usd=300000,
            market_value_usd=305000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Switzerland"],
            container_type="REEFER",
            description="Amoxicillin trihydrate API, pharmaceutical grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-008",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0009",
            port_of_entry="Kolkata",
            importer_name="Hindustan Unilever Ltd",
            iec_code="IEC0407067890",
            iec_age_months=240,
            country_of_origin="Malaysia",
            country_of_export="Malaysia",
            commodity="Palm Oil",
            hs_code="1511.10",
            declared_weight_kg=20000,
            declared_value_usd=22000,
            market_value_usd=22500,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Malaysia"],
            container_type="TANK",
            description="Crude palm oil, FFA 5% max",
        ),
        metadata=CaseMetadata(
            case_id="CASE-009",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0010",
            port_of_entry="Mundra",
            importer_name="ONGC Petro Additions Ltd",
            iec_code="IEC0712078901",
            iec_age_months=72,
            country_of_origin="UAE",
            country_of_export="UAE",
            commodity="Polyethylene Granules",
            hs_code="3901.10",
            declared_weight_kg=80000,
            declared_value_usd=96000,
            market_value_usd=97000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["UAE"],
            container_type="40GP",
            description="HDPE granules, density 0.955 g/cc",
        ),
        metadata=CaseMetadata(
            case_id="CASE-010",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0011",
            port_of_entry="JNPT",
            importer_name="Bajaj Electricals Ltd",
            iec_code="IEC0293089012",
            iec_age_months=156,
            country_of_origin="Taiwan",
            country_of_export="Taiwan",
            commodity="LED Driver ICs",
            hs_code="8542.31",
            declared_weight_kg=200,
            declared_value_usd=80000,
            market_value_usd=82000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Taiwan"],
            container_type="20GP",
            description="LED driver integrated circuits, surface mount",
        ),
        metadata=CaseMetadata(
            case_id="CASE-011",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0012",
            port_of_entry="Chennai",
            importer_name="TVS Motor Company Ltd",
            iec_code="IEC0419090123",
            iec_age_months=168,
            country_of_origin="Thailand",
            country_of_export="Thailand",
            commodity="Rubber Sheets",
            hs_code="4001.21",
            declared_weight_kg=12000,
            declared_value_usd=18000,
            market_value_usd=18500,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Thailand"],
            container_type="20GP",
            description="Ribbed smoked sheet rubber, RSS3 grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-012",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0013",
            port_of_entry="Kolkata",
            importer_name="Emami Ltd",
            iec_code="IEC0534101234",
            iec_age_months=90,
            country_of_origin="Indonesia",
            country_of_export="Indonesia",
            commodity="Coconut Oil",
            hs_code="1513.11",
            declared_weight_kg=15000,
            declared_value_usd=19500,
            market_value_usd=20000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Indonesia"],
            container_type="TANK",
            description="Virgin coconut oil, food grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-013",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0014",
            port_of_entry="JNPT",
            importer_name="Larsen & Toubro Ltd",
            iec_code="IEC0648112345",
            iec_age_months=204,
            country_of_origin="USA",
            country_of_export="USA",
            commodity="Industrial Pumps",
            hs_code="8413.70",
            declared_weight_kg=6000,
            declared_value_usd=240000,
            market_value_usd=245000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["USA"],
            container_type="40GP",
            description="Centrifugal pumps for oil refinery applications",
        ),
        metadata=CaseMetadata(
            case_id="CASE-014",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0015",
            port_of_entry="Mundra",
            importer_name="Asian Paints Ltd",
            iec_code="IEC0763123456",
            iec_age_months=78,
            country_of_origin="Germany",
            country_of_export="Germany",
            commodity="Titanium Dioxide Pigment",
            hs_code="3206.11",
            declared_weight_kg=22000,
            declared_value_usd=55000,
            market_value_usd=56000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Germany"],
            container_type="20GP",
            description="TiO2 rutile grade, surface treated",
        ),
        metadata=CaseMetadata(
            case_id="CASE-015",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0016",
            port_of_entry="Chennai",
            importer_name="Sundram Fasteners Ltd",
            iec_code="IEC0877134567",
            iec_age_months=192,
            country_of_origin="Japan",
            country_of_export="Japan",
            commodity="Steel Wire Rods",
            hs_code="7213.91",
            declared_weight_kg=50000,
            declared_value_usd=37500,
            market_value_usd=38000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Japan"],
            container_type="40GP",
            description="High carbon steel wire rods, 5.5mm diameter",
        ),
        metadata=CaseMetadata(
            case_id="CASE-016",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0017",
            port_of_entry="Kolkata",
            importer_name="Dabur India Ltd",
            iec_code="IEC0391145678",
            iec_age_months=216,
            country_of_origin="Nepal",
            country_of_export="Nepal",
            commodity="Herbs and Botanicals",
            hs_code="1211.90",
            declared_weight_kg=3000,
            declared_value_usd=9000,
            market_value_usd=9200,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Nepal"],
            container_type="20GP",
            description="Dried ayurvedic herbs, ashwagandha roots",
        ),
        metadata=CaseMetadata(
            case_id="CASE-017",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0018",
            port_of_entry="JNPT",
            importer_name="Godrej Consumer Products Ltd",
            iec_code="IEC0505156789",
            iec_age_months=132,
            country_of_origin="France",
            country_of_export="France",
            commodity="Fragrance Compounds",
            hs_code="3302.10",
            declared_weight_kg=800,
            declared_value_usd=160000,
            market_value_usd=163000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["France"],
            container_type="20GP",
            description="Aroma chemical blends for personal care",
        ),
        metadata=CaseMetadata(
            case_id="CASE-018",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0019",
            port_of_entry="Mundra",
            importer_name="Ultratech Cement Ltd",
            iec_code="IEC0619167890",
            iec_age_months=144,
            country_of_origin="UAE",
            country_of_export="UAE",
            commodity="Clinker",
            hs_code="2523.10",
            declared_weight_kg=300000,
            declared_value_usd=36000,
            market_value_usd=37000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["UAE"],
            container_type="BULK",
            description="Portland cement clinker, C3S 58%",
        ),
        metadata=CaseMetadata(
            case_id="CASE-019",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0020",
            port_of_entry="Chennai",
            importer_name="MRF Ltd",
            iec_code="IEC0733178901",
            iec_age_months=168,
            country_of_origin="Vietnam",
            country_of_export="Vietnam",
            commodity="Natural Latex",
            hs_code="4001.10",
            declared_weight_kg=18000,
            declared_value_usd=27000,
            market_value_usd=27500,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Vietnam"],
            container_type="TANK",
            description="Field latex, 60% dry rubber content",
        ),
        metadata=CaseMetadata(
            case_id="CASE-020",
            difficulty="clean",
            true_anomalies=[],
            correct_channel=Channel.GREEN,
        ),
    ),

    # =========================================================
    # EASY CASES (20) — 1 anomaly, correct channel = ORANGE/RED
    # =========================================================

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0021",
            port_of_entry="JNPT",
            importer_name="Sunshine Trading Co",
            iec_code="IEC0847190012",
            iec_age_months=6,           # new IEC
            country_of_origin="USA",
            country_of_export="USA",
            commodity="Medical Equipment",
            hs_code="9018.90",
            declared_weight_kg=2000,
            declared_value_usd=1500000, # high value first shipment
            market_value_usd=1520000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["USA"],
            container_type="20GP",
            description="Surgical robots and diagnostic imaging equipment",
        ),
        metadata=CaseMetadata(
            case_id="CASE-021",
            difficulty="easy",
            true_anomalies=[AnomalyType.NEW_IEC_HIGH_VALUE],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0022",
            port_of_entry="Mundra",
            importer_name="Global Gems Pvt Ltd",
            iec_code="IEC0961201234",
            iec_age_months=48,
            country_of_origin="Myanmar",   # high risk FATF grey list
            country_of_export="Myanmar",
            commodity="Ruby Gemstones",
            hs_code="7103.91",
            declared_weight_kg=5,
            declared_value_usd=50000,
            market_value_usd=52000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Myanmar"],
            container_type="20GP",
            description="Natural ruby gemstones, uncut",
        ),
        metadata=CaseMetadata(
            case_id="CASE-022",
            difficulty="easy",
            true_anomalies=[AnomalyType.HIGH_RISK_ORIGIN],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0023",
            port_of_entry="Chennai",
            importer_name="Delta Metals Ltd",
            iec_code="IEC0275212345",
            iec_age_months=60,
            country_of_origin="China",
            country_of_export="China",
            commodity="Stainless Steel Pipes",
            hs_code="7304.41",          # CBIC flagged HS code for misdeclaration
            declared_weight_kg=30000,
            declared_value_usd=42000,
            market_value_usd=43000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China"],
            container_type="40GP",
            description="SS 304 seamless pipes, schedule 40",
        ),
        metadata=CaseMetadata(
            case_id="CASE-023",
            difficulty="easy",
            true_anomalies=[AnomalyType.HS_CODE_RISK],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=500000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0024",
            port_of_entry="Kolkata",
            importer_name="Rapid Commerce Pvt Ltd",
            iec_code="IEC0389223456",
            iec_age_months=72,
            country_of_origin="Bangladesh",
            country_of_export="Bangladesh",
            commodity="Ready-made Garments",
            hs_code="6203.42",
            declared_weight_kg=8000,
            declared_value_usd=4000,    # severely undervalued, ~$0.5/kg (market ~$5/kg)
            market_value_usd=40000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Bangladesh"],
            container_type="20GP",
            description="Men's denim trousers, export quality",
        ),
        metadata=CaseMetadata(
            case_id="CASE-024",
            difficulty="easy",
            true_anomalies=[AnomalyType.SEVERE_UNDERVALUATION],
            correct_channel=Channel.RED,
            duty_gap_inr=2800000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0025",
            port_of_entry="JNPT",
            importer_name="Vikram Chemicals Ltd",
            iec_code="IEC0497234567",
            iec_age_months=108,
            country_of_origin="China",
            country_of_export="Hong Kong",  # suspicious: routed via HK unnecessarily
            commodity="Industrial Chemicals",
            hs_code="2903.19",
            declared_weight_kg=5000,
            declared_value_usd=25000,
            market_value_usd=26000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China", "Hong Kong"],
            container_type="20GP",
            description="Chlorinated solvents, technical grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-025",
            difficulty="easy",
            true_anomalies=[AnomalyType.SUSPICIOUS_ROUTING],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0026",
            port_of_entry="Mundra",
            importer_name="Premier Textiles Ltd",
            iec_code="IEC0611245678",
            iec_age_months=84,
            country_of_origin="Pakistan",
            country_of_export="Pakistan",
            commodity="Cotton Yarn",
            hs_code="5205.11",
            declared_weight_kg=20000,
            declared_value_usd=16000,
            market_value_usd=40000,     # declared at 40% of market price
            previous_violations=0,
            related_party=True,
            related_party_disclosed=False,  # undisclosed!
            routing_countries=["Pakistan"],
            container_type="20GP",
            description="Combed cotton yarn, 20s count",
        ),
        metadata=CaseMetadata(
            case_id="CASE-026",
            difficulty="easy",
            true_anomalies=[AnomalyType.UNDISCLOSED_RELATED_PARTY],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=1200000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0027",
            port_of_entry="Chennai",
            importer_name="Apex Electronics Pvt Ltd",
            iec_code="IEC0725256789",
            iec_age_months=96,
            country_of_origin="China",
            country_of_export="China",
            commodity="Mobile Phone Components",
            hs_code="8517.70",
            declared_weight_kg=500,
            declared_value_usd=2500,    # severely undervalued (market $250K+)
            market_value_usd=250000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China"],
            container_type="20GP",
            description="Smartphone OLED display assemblies, 6.1 inch",
        ),
        metadata=CaseMetadata(
            case_id="CASE-027",
            difficulty="easy",
            true_anomalies=[AnomalyType.SEVERE_UNDERVALUATION],
            correct_channel=Channel.RED,
            duty_gap_inr=18000000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0028",
            port_of_entry="Kolkata",
            importer_name="North Star Impex",
            iec_code="IEC0839267890",
            iec_age_months=120,
            country_of_origin="China",
            country_of_export="China",
            commodity="Steel Billets",
            hs_code="7207.11",
            declared_weight_kg=200000,  # 200 tons in a 20GP — impossible (capacity ~18T)
            declared_value_usd=120000,
            market_value_usd=None,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China"],
            container_type="20GP",
            description="Carbon steel billets, 100x100mm",
        ),
        metadata=CaseMetadata(
            case_id="CASE-028",
            difficulty="easy",
            true_anomalies=[AnomalyType.WEIGHT_VOLUME_MISMATCH],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0029",
            port_of_entry="JNPT",
            importer_name="Falcon Trading Co",
            iec_code="IEC0953278901",
            iec_age_months=36,
            country_of_origin="Iran",   # OFAC sanctioned
            country_of_export="Oman",
            commodity="Petrochemical Products",
            hs_code="3902.10",
            declared_weight_kg=40000,
            declared_value_usd=48000,
            market_value_usd=50000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Iran", "Oman"],
            container_type="40GP",
            description="Polypropylene granules, injection moulding grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-029",
            difficulty="easy",
            true_anomalies=[AnomalyType.HIGH_RISK_ORIGIN],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0030",
            port_of_entry="Mundra",
            importer_name="Krishna Traders",
            iec_code="IEC0367289012",
            iec_age_months=144,
            country_of_origin="China",
            country_of_export="Dubai",     # routed India-bound goods via Dubai unnecessarily
            commodity="Brass Fittings",
            hs_code="7412.20",             # CBIC flagged HSN for antidumping
            declared_weight_kg=10000,
            declared_value_usd=35000,
            market_value_usd=36000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China", "UAE", "Dubai"],
            container_type="20GP",
            description="Brass compression fittings, various sizes",
        ),
        metadata=CaseMetadata(
            case_id="CASE-030",
            difficulty="easy",
            true_anomalies=[AnomalyType.HS_CODE_RISK],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=300000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0031",
            port_of_entry="Chennai",
            importer_name="Metro Footwear Pvt Ltd",
            iec_code="IEC0481290123",
            iec_age_months=60,
            country_of_origin="China",
            country_of_export="China",
            commodity="Leather Footwear",
            hs_code="6403.99",
            declared_weight_kg=5000,
            declared_value_usd=3000,     # $0.6/pair vs market ~$8/pair
            market_value_usd=40000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China"],
            container_type="20GP",
            description="Men's leather dress shoes, size 6-11",
        ),
        metadata=CaseMetadata(
            case_id="CASE-031",
            difficulty="easy",
            true_anomalies=[AnomalyType.SEVERE_UNDERVALUATION],
            correct_channel=Channel.RED,
            duty_gap_inr=2500000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0032",
            port_of_entry="Kolkata",
            importer_name="Eastgate Imports",
            iec_code="IEC0595301234",
            iec_age_months=2,            # very new IEC
            country_of_origin="UK",
            country_of_export="UK",
            commodity="Luxury Watches",
            hs_code="9102.11",
            declared_weight_kg=50,
            declared_value_usd=2000000,  # very high value first shipment
            market_value_usd=2050000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["UK"],
            container_type="20GP",
            description="Swiss luxury watches, mechanical movement",
        ),
        metadata=CaseMetadata(
            case_id="CASE-032",
            difficulty="easy",
            true_anomalies=[AnomalyType.NEW_IEC_HIGH_VALUE],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0033",
            port_of_entry="JNPT",
            importer_name="Dynamic Steels Ltd",
            iec_code="IEC0709312345",
            iec_age_months=96,
            country_of_origin="South Korea",
            country_of_export="South Korea",
            commodity="Hot Rolled Coils",
            hs_code="7208.36",
            declared_weight_kg=500000,   # 500T in single 20GP — impossible
            declared_value_usd=175000,
            market_value_usd=180000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["South Korea"],
            container_type="20GP",
            description="HR coils, 2.5mm thickness, Grade A",
        ),
        metadata=CaseMetadata(
            case_id="CASE-033",
            difficulty="easy",
            true_anomalies=[AnomalyType.WEIGHT_VOLUME_MISMATCH],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0034",
            port_of_entry="Mundra",
            importer_name="Prism Plastics Pvt Ltd",
            iec_code="IEC0823323456",
            iec_age_months=120,
            country_of_origin="China",
            country_of_export="Vietnam",   # China goods via Vietnam — suspicious
            commodity="PVC Pipes",
            hs_code="3917.22",
            declared_weight_kg=25000,
            declared_value_usd=30000,
            market_value_usd=31000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China", "Vietnam"],
            container_type="40GP",
            description="PVC pressure pipes, 110mm diameter",
        ),
        metadata=CaseMetadata(
            case_id="CASE-034",
            difficulty="easy",
            true_anomalies=[AnomalyType.SUSPICIOUS_ROUTING],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0035",
            port_of_entry="Chennai",
            importer_name="Omega Commodities",
            iec_code="IEC0937334567",
            iec_age_months=60,
            country_of_origin="Turkey",
            country_of_export="Turkey",
            commodity="Marble Slabs",
            hs_code="6802.92",
            related_party=True,
            related_party_disclosed=False,   # undisclosed related party
            declared_weight_kg=80000,
            declared_value_usd=64000,
            market_value_usd=66000,
            previous_violations=0,
            routing_countries=["Turkey"],
            container_type="40GP",
            description="White Carrara marble slabs, 2cm thickness",
        ),
        metadata=CaseMetadata(
            case_id="CASE-035",
            difficulty="easy",
            true_anomalies=[AnomalyType.UNDISCLOSED_RELATED_PARTY],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0036",
            port_of_entry="Kolkata",
            importer_name="Bengal Agro Products",
            iec_code="IEC0451345678",
            iec_age_months=72,
            country_of_origin="Myanmar",   # FATF grey list
            country_of_export="Myanmar",
            commodity="Teak Wood Logs",
            hs_code="4403.49",
            declared_weight_kg=30000,
            declared_value_usd=45000,
            market_value_usd=46000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Myanmar"],
            container_type="40GP",
            description="Teak logs, 30-50cm diameter",
        ),
        metadata=CaseMetadata(
            case_id="CASE-036",
            difficulty="easy",
            true_anomalies=[AnomalyType.HIGH_RISK_ORIGIN],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0037",
            port_of_entry="JNPT",
            importer_name="Silver Electronics",
            iec_code="IEC0565356789",
            iec_age_months=84,
            country_of_origin="China",
            country_of_export="China",
            commodity="Television Sets",
            hs_code="8528.72",            # CBIC flagged HS for undervaluation
            declared_weight_kg=12000,
            declared_value_usd=36000,
            market_value_usd=37000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China"],
            container_type="40HC",
            description="55-inch 4K Smart TVs",
        ),
        metadata=CaseMetadata(
            case_id="CASE-037",
            difficulty="easy",
            true_anomalies=[AnomalyType.HS_CODE_RISK],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=200000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0038",
            port_of_entry="Mundra",
            importer_name="QuickStart Ventures",
            iec_code="IEC0679367890",
            iec_age_months=3,             # very new IEC
            country_of_origin="Germany",
            country_of_export="Germany",
            commodity="Industrial Machinery",
            hs_code="8479.89",
            declared_weight_kg=8000,
            declared_value_usd=800000,    # very high value, new IEC
            market_value_usd=810000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Germany"],
            container_type="40HC",
            description="Bottling line machinery, automated",
        ),
        metadata=CaseMetadata(
            case_id="CASE-038",
            difficulty="easy",
            true_anomalies=[AnomalyType.NEW_IEC_HIGH_VALUE],
            correct_channel=Channel.ORANGE,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0039",
            port_of_entry="Chennai",
            importer_name="Coastal Fish Exports",
            iec_code="IEC0793378901",
            iec_age_months=96,
            country_of_origin="Sri Lanka",
            country_of_export="Sri Lanka",
            commodity="Frozen Tuna",
            hs_code="0303.46",
            declared_weight_kg=150000,    # 150T in a 20GP reefer (capacity ~25T)
            declared_value_usd=300000,
            market_value_usd=None,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Sri Lanka"],
            container_type="20RF",
            description="Frozen yellowfin tuna, whole, H&G",
        ),
        metadata=CaseMetadata(
            case_id="CASE-039",
            difficulty="easy",
            true_anomalies=[AnomalyType.WEIGHT_VOLUME_MISMATCH],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0040",
            port_of_entry="Kolkata",
            importer_name="Royal Spice Traders",
            iec_code="IEC0407389012",
            iec_age_months=48,
            country_of_origin="Afghanistan",  # FATF grey list
            country_of_export="Afghanistan",
            commodity="Dry Fruits",
            hs_code="0813.50",
            declared_weight_kg=10000,
            declared_value_usd=80000,
            market_value_usd=82000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Afghanistan"],
            container_type="20GP",
            description="Premium pine nuts and almonds",
        ),
        metadata=CaseMetadata(
            case_id="CASE-040",
            difficulty="easy",
            true_anomalies=[AnomalyType.HIGH_RISK_ORIGIN],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    # ===========================================================
    # MEDIUM CASES (12) — 2 anomalies, correct channel = RED
    # ===========================================================

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0041",
            port_of_entry="JNPT",
            importer_name="National Steel Pvt Ltd",
            iec_code="IEC0521400123",
            iec_age_months=60,
            country_of_origin="China",
            country_of_export="Thailand",   # suspicious: China goods via Thailand
            commodity="Flat Rolled Steel",
            hs_code="7209.16",
            declared_weight_kg=60000,
            declared_value_usd=18000,       # severely undervalued ($0.30/kg, market $0.70/kg)
            market_value_usd=42000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China", "Thailand"],
            container_type="40GP",
            description="Cold rolled steel coils, 0.5mm, CS Grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-041",
            difficulty="medium",
            true_anomalies=[AnomalyType.SUSPICIOUS_ROUTING, AnomalyType.SEVERE_UNDERVALUATION],
            correct_channel=Channel.RED,
            duty_gap_inr=1800000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0042",
            port_of_entry="Mundra",
            importer_name="Zain Import House",
            iec_code="IEC0635411234",
            iec_age_months=4,             # new IEC
            country_of_origin="UAE",
            country_of_export="UAE",
            commodity="Gold Jewellery",
            hs_code="7113.19",
            declared_weight_kg=100,
            declared_value_usd=3500000,   # very high value + new IEC
            market_value_usd=3550000,
            previous_violations=0,
            related_party=True,
            related_party_disclosed=False,  # undisclosed related party
            routing_countries=["UAE"],
            container_type="20GP",
            description="22-karat gold jewellery sets",
        ),
        metadata=CaseMetadata(
            case_id="CASE-042",
            difficulty="medium",
            true_anomalies=[AnomalyType.NEW_IEC_HIGH_VALUE, AnomalyType.UNDISCLOSED_RELATED_PARTY],
            correct_channel=Channel.RED,
            duty_gap_inr=5000000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0043",
            port_of_entry="Chennai",
            importer_name="Sunflower Electronics",
            iec_code="IEC0749422345",
            iec_age_months=72,
            country_of_origin="North Korea",  # OFAC sanctioned
            country_of_export="China",
            commodity="Electronic Components",
            hs_code="8542.39",              # flagged HS code
            declared_weight_kg=500,
            declared_value_usd=50000,
            market_value_usd=52000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["North Korea", "China"],
            container_type="20GP",
            description="DRAM memory chips, 8GB DDR4",
        ),
        metadata=CaseMetadata(
            case_id="CASE-043",
            difficulty="medium",
            true_anomalies=[AnomalyType.HIGH_RISK_ORIGIN, AnomalyType.HS_CODE_RISK],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0044",
            port_of_entry="Kolkata",
            importer_name="Bengal Textile Mills",
            iec_code="IEC0463433456",
            iec_age_months=96,
            country_of_origin="Bangladesh",
            country_of_export="Bangladesh",
            commodity="Synthetic Fabric",
            hs_code="5407.61",
            declared_weight_kg=200000,     # 200T in single 20GP — impossible
            declared_value_usd=60000,       # also undervalued
            market_value_usd=300000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Bangladesh"],
            container_type="20GP",
            description="Polyester woven fabric, 100gsm",
        ),
        metadata=CaseMetadata(
            case_id="CASE-044",
            difficulty="medium",
            true_anomalies=[AnomalyType.WEIGHT_VOLUME_MISMATCH, AnomalyType.SEVERE_UNDERVALUATION],
            correct_channel=Channel.RED,
            duty_gap_inr=15000000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0045",
            port_of_entry="JNPT",
            importer_name="Phoenix Chemicals",
            iec_code="IEC0577444567",
            iec_age_months=108,
            country_of_origin="Syria",    # OFAC sanctioned
            country_of_export="Turkey",
            commodity="Industrial Solvents",
            hs_code="2901.10",
            declared_weight_kg=20000,
            declared_value_usd=8000,       # severely undervalued
            market_value_usd=40000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Syria", "Turkey"],
            container_type="TANK",
            description="Aliphatic hydrocarbons, C9-C12 fraction",
        ),
        metadata=CaseMetadata(
            case_id="CASE-045",
            difficulty="medium",
            true_anomalies=[AnomalyType.HIGH_RISK_ORIGIN, AnomalyType.SEVERE_UNDERVALUATION],
            correct_channel=Channel.RED,
            duty_gap_inr=2400000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0046",
            port_of_entry="Mundra",
            importer_name="Titan Traders Pvt Ltd",
            iec_code="IEC0691455678",
            iec_age_months=120,
            country_of_origin="China",
            country_of_export="Malaysia",   # China goods via Malaysia to avoid antidumping
            commodity="Aluminium Extrusions",
            hs_code="7604.21",             # flagged for antidumping origin fraud
            declared_weight_kg=40000,
            declared_value_usd=80000,
            market_value_usd=82000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China", "Malaysia"],
            container_type="40GP",
            description="Aluminium alloy extrusion profiles, 6063-T5",
        ),
        metadata=CaseMetadata(
            case_id="CASE-046",
            difficulty="medium",
            true_anomalies=[AnomalyType.SUSPICIOUS_ROUTING, AnomalyType.HS_CODE_RISK],
            correct_channel=Channel.RED,
            duty_gap_inr=1000000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0047",
            port_of_entry="Chennai",
            importer_name="VIP Garments Pvt Ltd",
            iec_code="IEC0805466789",
            iec_age_months=84,
            country_of_origin="China",
            country_of_export="China",
            commodity="Branded Clothing",
            hs_code="6110.20",
            declared_weight_kg=10000,
            declared_value_usd=5000,       # severely undervalued
            market_value_usd=80000,
            previous_violations=0,
            related_party=True,
            related_party_disclosed=False,  # undisclosed
            routing_countries=["China"],
            container_type="40GP",
            description="Men's cotton T-shirts, international brand",
        ),
        metadata=CaseMetadata(
            case_id="CASE-047",
            difficulty="medium",
            true_anomalies=[AnomalyType.SEVERE_UNDERVALUATION, AnomalyType.UNDISCLOSED_RELATED_PARTY],
            correct_channel=Channel.RED,
            duty_gap_inr=5600000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0048",
            port_of_entry="Kolkata",
            importer_name="NewEdge Imports",
            iec_code="IEC0319477890",
            iec_age_months=5,             # new IEC
            country_of_origin="UAE",
            country_of_export="UAE",
            commodity="Cut Diamonds",
            hs_code="7102.39",
            declared_weight_kg=2,
            declared_value_usd=1200000,   # high value first shipment
            market_value_usd=1230000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["UAE", "Belgium"],  # Belgium to UAE to India — suspicious
            container_type="20GP",
            description="Polished diamonds, 0.5-1.0 carat, VVS1",
        ),
        metadata=CaseMetadata(
            case_id="CASE-048",
            difficulty="medium",
            true_anomalies=[AnomalyType.NEW_IEC_HIGH_VALUE, AnomalyType.SUSPICIOUS_ROUTING],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0049",
            port_of_entry="JNPT",
            importer_name="Continental Goods Ltd",
            iec_code="IEC0433488901",
            iec_age_months=96,
            country_of_origin="China",
            country_of_export="China",
            commodity="Copper Tubes",
            hs_code="7411.10",
            declared_weight_kg=1000000,   # 1000T in single 20GP — absurd
            declared_value_usd=800000,
            market_value_usd=None,
            previous_violations=0,
            related_party=True,
            related_party_disclosed=False,  # undisclosed
            routing_countries=["China"],
            container_type="20GP",
            description="Copper refrigeration tubes, 12mm OD",
        ),
        metadata=CaseMetadata(
            case_id="CASE-049",
            difficulty="medium",
            true_anomalies=[AnomalyType.WEIGHT_VOLUME_MISMATCH, AnomalyType.UNDISCLOSED_RELATED_PARTY],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0050",
            port_of_entry="Mundra",
            importer_name="Classic Impex",
            iec_code="IEC0547499012",
            iec_age_months=120,
            country_of_origin="North Korea",  # OFAC sanctioned
            country_of_export="China",
            commodity="Coal",
            hs_code="2701.12",
            declared_weight_kg=80000,
            declared_value_usd=4000,       # severely undervalued ($0.05/kg, market $0.14/kg)
            market_value_usd=11200,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["North Korea", "China"],
            container_type="BULK",
            description="Coking coal, ash 8%, moisture 10%",
        ),
        metadata=CaseMetadata(
            case_id="CASE-050",
            difficulty="medium",
            true_anomalies=[AnomalyType.HIGH_RISK_ORIGIN, AnomalyType.SEVERE_UNDERVALUATION],
            correct_channel=Channel.RED,
            duty_gap_inr=600000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0051",
            port_of_entry="Chennai",
            importer_name="Velocity Commerce",
            iec_code="IEC0661500123",
            iec_age_months=2,             # new IEC
            country_of_origin="China",
            country_of_export="Singapore", # suspicious routing
            commodity="Electronic Toys",
            hs_code="9503.00",
            declared_weight_kg=3000,
            declared_value_usd=900000,    # high value first shipment
            market_value_usd=920000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China", "Singapore"],
            container_type="40HC",
            description="Remote controlled toys and drones",
        ),
        metadata=CaseMetadata(
            case_id="CASE-051",
            difficulty="medium",
            true_anomalies=[AnomalyType.NEW_IEC_HIGH_VALUE, AnomalyType.SUSPICIOUS_ROUTING],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0052",
            port_of_entry="Kolkata",
            importer_name="Heritage Handicrafts",
            iec_code="IEC0775511234",
            iec_age_months=84,
            country_of_origin="Myanmar",   # FATF grey list
            country_of_export="Myanmar",
            commodity="Jade Stones",
            hs_code="7103.10",             # flagged HS for misdeclaration
            declared_weight_kg=500,
            declared_value_usd=100000,
            market_value_usd=103000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Myanmar"],
            container_type="20GP",
            description="Natural jade boulders and rough stones",
        ),
        metadata=CaseMetadata(
            case_id="CASE-052",
            difficulty="medium",
            true_anomalies=[AnomalyType.HIGH_RISK_ORIGIN, AnomalyType.HS_CODE_RISK],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    # ===========================================================
    # HARD CASES (8) — 3+ anomalies, always RED
    # ===========================================================

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0053",
            port_of_entry="JNPT",
            importer_name="Nexus Trade Links",
            iec_code="IEC0889522345",
            iec_age_months=36,
            country_of_origin="Iran",      # OFAC sanctioned
            country_of_export="Turkey",    # suspicious routing
            commodity="Petrochemicals",
            hs_code="2902.20",             # flagged HS
            declared_weight_kg=50000,
            declared_value_usd=10000,      # severely undervalued
            market_value_usd=100000,
            previous_violations=3,         # repeat violator
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Iran", "Turkey"],
            container_type="TANK",
            description="Benzene, chemical grade 99.5%",
        ),
        metadata=CaseMetadata(
            case_id="CASE-053",
            difficulty="hard",
            true_anomalies=[
                AnomalyType.REPEAT_VIOLATOR,
                AnomalyType.HIGH_RISK_ORIGIN,
                AnomalyType.SUSPICIOUS_ROUTING,
                AnomalyType.SEVERE_UNDERVALUATION,
            ],
            correct_channel=Channel.RED,
            duty_gap_inr=6600000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0054",
            port_of_entry="Mundra",
            importer_name="Crescent Metals",
            iec_code="IEC0403533456",
            iec_age_months=48,
            country_of_origin="China",
            country_of_export="Malaysia",   # suspicious (antidumping bypass)
            commodity="Steel Angles",
            hs_code="7216.31",             # flagged for origin fraud
            declared_weight_kg=400000,      # 400T in 20GP — impossible
            declared_value_usd=40000,       # severely undervalued
            market_value_usd=200000,
            previous_violations=2,          # repeat violator
            related_party=True,
            related_party_disclosed=False,  # undisclosed
            routing_countries=["China", "Malaysia"],
            container_type="20GP",
            description="MS angle sections, 75x75mm",
        ),
        metadata=CaseMetadata(
            case_id="CASE-054",
            difficulty="hard",
            true_anomalies=[
                AnomalyType.REPEAT_VIOLATOR,
                AnomalyType.SUSPICIOUS_ROUTING,
                AnomalyType.WEIGHT_VOLUME_MISMATCH,
                AnomalyType.SEVERE_UNDERVALUATION,
                AnomalyType.UNDISCLOSED_RELATED_PARTY,
            ],
            correct_channel=Channel.RED,
            duty_gap_inr=12000000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0055",
            port_of_entry="Chennai",
            importer_name="Alpha Commodities",
            iec_code="IEC0517544567",
            iec_age_months=1,              # brand new IEC
            country_of_origin="Afghanistan", # FATF grey list
            country_of_export="Pakistan",   # suspicious routing
            commodity="Raw Opium Derivatives",
            hs_code="2939.11",
            declared_weight_kg=100,
            declared_value_usd=5000000,    # extremely high value first shipment
            market_value_usd=5100000,
            previous_violations=0,
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["Afghanistan", "Pakistan"],
            container_type="20GP",
            description="Pharmaceutical opioid precursors",
        ),
        metadata=CaseMetadata(
            case_id="CASE-055",
            difficulty="hard",
            true_anomalies=[
                AnomalyType.NEW_IEC_HIGH_VALUE,
                AnomalyType.HIGH_RISK_ORIGIN,
                AnomalyType.SUSPICIOUS_ROUTING,
            ],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0056",
            port_of_entry="Kolkata",
            importer_name="Royal Trading Corp",
            iec_code="IEC0631555678",
            iec_age_months=60,
            country_of_origin="China",
            country_of_export="Hong Kong",  # suspicious
            commodity="Smartphones",
            hs_code="8517.12",              # flagged HS for undervaluation
            declared_weight_kg=1000,
            declared_value_usd=10000,       # declared at $10/unit, market ~$300/unit
            market_value_usd=300000,
            previous_violations=4,          # severe repeat violator
            related_party=True,
            related_party_disclosed=False,  # undisclosed
            routing_countries=["China", "Hong Kong"],
            container_type="20GP",
            description="Budget Android smartphones, 64GB storage",
        ),
        metadata=CaseMetadata(
            case_id="CASE-056",
            difficulty="hard",
            true_anomalies=[
                AnomalyType.REPEAT_VIOLATOR,
                AnomalyType.SUSPICIOUS_ROUTING,
                AnomalyType.SEVERE_UNDERVALUATION,
                AnomalyType.UNDISCLOSED_RELATED_PARTY,
                AnomalyType.HS_CODE_RISK,
            ],
            correct_channel=Channel.RED,
            duty_gap_inr=22000000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-JNPT-2024-0057",
            port_of_entry="JNPT",
            importer_name="Spectrum Chemicals",
            iec_code="IEC0745566789",
            iec_age_months=84,
            country_of_origin="North Korea",  # OFAC sanctioned
            country_of_export="China",
            commodity="Chemical Precursors",
            hs_code="2921.41",              # flagged HS
            declared_weight_kg=5000,
            declared_value_usd=2000,         # severely undervalued
            market_value_usd=50000,
            previous_violations=2,           # repeat violator
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["North Korea", "China"],
            container_type="20GP",
            description="Aniline, industrial grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-057",
            difficulty="hard",
            true_anomalies=[
                AnomalyType.REPEAT_VIOLATOR,
                AnomalyType.HIGH_RISK_ORIGIN,
                AnomalyType.SEVERE_UNDERVALUATION,
                AnomalyType.HS_CODE_RISK,
            ],
            correct_channel=Channel.RED,
            duty_gap_inr=3600000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-MUNDRA-2024-0058",
            port_of_entry="Mundra",
            importer_name="Sunrise Imports",
            iec_code="IEC0859577890",
            iec_age_months=2,              # new IEC
            country_of_origin="Myanmar",   # FATF grey list
            country_of_export="Thailand",  # suspicious routing
            commodity="Precious Stones",
            hs_code="7103.99",             # flagged HS
            declared_weight_kg=50,
            declared_value_usd=2500000,    # high value first shipment
            market_value_usd=2550000,
            previous_violations=0,
            related_party=True,
            related_party_disclosed=False,  # undisclosed
            routing_countries=["Myanmar", "Thailand"],
            container_type="20GP",
            description="Mixed gemstones: sapphires, rubies, emeralds",
        ),
        metadata=CaseMetadata(
            case_id="CASE-058",
            difficulty="hard",
            true_anomalies=[
                AnomalyType.NEW_IEC_HIGH_VALUE,
                AnomalyType.HIGH_RISK_ORIGIN,
                AnomalyType.SUSPICIOUS_ROUTING,
                AnomalyType.UNDISCLOSED_RELATED_PARTY,
                AnomalyType.HS_CODE_RISK,
            ],
            correct_channel=Channel.RED,
            duty_gap_inr=None,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-CHENNAI-2024-0059",
            port_of_entry="Chennai",
            importer_name="Horizon Electronics",
            iec_code="IEC0473588901",
            iec_age_months=120,
            country_of_origin="China",
            country_of_export="Vietnam",   # suspicious (China via Vietnam)
            commodity="Integrated Circuits",
            hs_code="8542.39",             # flagged HS
            declared_weight_kg=200000,      # 200T in 20GP — impossible
            declared_value_usd=5000,        # severely undervalued
            market_value_usd=2000000,
            previous_violations=3,          # repeat violator
            related_party=False,
            related_party_disclosed=True,
            routing_countries=["China", "Vietnam"],
            container_type="20GP",
            description="High-density FPGA chips, commercial grade",
        ),
        metadata=CaseMetadata(
            case_id="CASE-059",
            difficulty="hard",
            true_anomalies=[
                AnomalyType.REPEAT_VIOLATOR,
                AnomalyType.SUSPICIOUS_ROUTING,
                AnomalyType.WEIGHT_VOLUME_MISMATCH,
                AnomalyType.SEVERE_UNDERVALUATION,
                AnomalyType.HS_CODE_RISK,
            ],
            correct_channel=Channel.RED,
            duty_gap_inr=148000000,
        ),
    ),

    CustomsCase(
        manifest=CargoManifest(
            boe_number="BOE-KOLKATA-2024-0060",
            port_of_entry="Kolkata",
            importer_name="Summit Trading Pvt Ltd",
            iec_code="IEC0587599012",
            iec_age_months=3,              # very new IEC
            country_of_origin="Iran",      # OFAC sanctioned
            country_of_export="Oman",      # suspicious (Iran via Oman)
            commodity="Crude Oil Derivatives",
            hs_code="2710.12",             # flagged HS
            declared_weight_kg=100000,
            declared_value_usd=10000,       # severely undervalued
            market_value_usd=80000,
            previous_violations=2,          # repeat violator
            related_party=True,
            related_party_disclosed=False,  # undisclosed
            routing_countries=["Iran", "Oman"],
            container_type="TANK",
            description="Light petroleum naphtha, C5-C8 fraction",
        ),
        metadata=CaseMetadata(
            case_id="CASE-060",
            difficulty="hard",
            true_anomalies=[
                AnomalyType.NEW_IEC_HIGH_VALUE,
                AnomalyType.REPEAT_VIOLATOR,
                AnomalyType.HIGH_RISK_ORIGIN,
                AnomalyType.SUSPICIOUS_ROUTING,
                AnomalyType.SEVERE_UNDERVALUATION,
                AnomalyType.UNDISCLOSED_RELATED_PARTY,
                AnomalyType.HS_CODE_RISK,
            ],
            correct_channel=Channel.RED,
            duty_gap_inr=5200000,
        ),
    ),
]

# Index by case_id for fast lookup
CASES_BY_ID: dict[str, CustomsCase] = {c.metadata.case_id: c for c in CASES}
