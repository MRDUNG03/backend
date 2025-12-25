from model.Connected_DB import connected_DB
def create_table_sensor():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ax FLOAT,
            ay FLOAT,
            az FLOAT,
            current FLOAT,
            voltage FLOAT,       
            temp FLOAT
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()