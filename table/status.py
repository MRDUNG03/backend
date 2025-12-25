from model.Connected_DB import connected_DB
def create_table_status():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS status(
            id INT PRIMARY KEY DEFAULT 1,
            status VARCHAR(50),   
            message TEXT,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        INSERT IGNORE INTO status (id, status, message) 
        VALUES (1, 'Normal', 'Hệ thống đang hoạt động bình thường')
    """)
    conn.commit()
    cursor.close()
    conn.close()
