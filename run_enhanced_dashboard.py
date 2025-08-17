#!/usr/bin/env python3
"""
Simple script to run the enhanced ShelfScale dashboard
"""

from shelfscale.visualization.enhanced_dashboard import create_enhanced_dashboard

if __name__ == "__main__":
    print("🚀 Starting Enhanced ShelfScale Dashboard...")
    print("📊 Loading food product database...")
    
    dashboard = create_enhanced_dashboard()
    
    print("✅ Dashboard ready!")
    print("🌐 Open your browser and go to: http://localhost:8050")
    print("📖 Click on food groups in the treemap to explore products")
    print("💾 Use download buttons to export data")
    print("\n🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        dashboard.run_server(debug=False, port=8050)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure all dependencies are installed and port 8050 is available")