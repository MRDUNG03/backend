
from model.connected_DB import connected_DB

def create_table_alerts():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device_id INT NOT NULL,
            alert VARCHAR(100) NOT NULL,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) DEFAULT 'new'
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()