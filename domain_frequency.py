import duckdb
from collections import Counter

conn = duckdb.connect()
domain_counts = Counter()

legal_keywords = ['law', 'legal', 'court', 'judicial', 'statute', 'attorney', 
                  'lawyer', 'litigation', 'jurisdiction', 'verdict', 'plaintiff',
                  'defendant', 'counsel', 'barrister', 'solicitor', 'jurisprudence', 'judge', 'justice']

keyword_filter = " OR ".join([f"url LIKE '%{k}%'" for k in legal_keywords])

for part in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]:
    fname = f"part-{part:05d}.parquet"
    print(f"Scanning {fname}...")
    result = conn.execute(f"""
        SELECT url_host_registered_domain, COUNT(*) as cnt
        FROM read_parquet('{fname}')
        GROUP BY 1
        ORDER BY cnt DESC
        LIMIT 50
    """).fetchall()
    for row in result:
        domain_counts[row[0]] += row[1]

print("\nTop 200 legal domains across all TLDs:")
for domain, count in domain_counts.most_common(200):
    print(f"{count:>8} {domain}")