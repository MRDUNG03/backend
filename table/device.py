from model.Connected_DB import connected_DB
def create_table_device():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device_name VARCHAR(100),
            status VARCHAR(50)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()