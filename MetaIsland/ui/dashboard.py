from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                            QTableWidget, QTableWidgetItem, QTextEdit, QListWidget, 
                            QLabel, QSplitter, QApplication)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.Qsci import QsciScintilla, QsciLexerPython
import sys
import json
import os

class IslandDashboard(QMainWindow):
    def __init__(self, execution_engine):
        super().__init__()
        self.execution_engine = execution_engine
        self.setWindowTitle("MetaIsland Simulation Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        self.setupUI()
        
        # Setup update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000)  # Update every 1 second

    def setupUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Agent status and metrics
        left_panel = QTabWidget()
        
        # Agent Status Tab
        self.agent_table = QTableWidget()
        self.agent_table.setColumnCount(6)
        self.agent_table.setHorizontalHeaderLabels([
            "ID", "Vitality", "Cargo", "Survival %", 
            "Last Perf.", "Code Versions"
        ])
        left_panel.addTab(self.agent_table, "Agent Status")
        
        # Mechanism Tab
        self.mechanism_list = QListWidget()
        left_panel.addTab(self.mechanism_list, "Mechanisms")
        
        # Right panel - Code and analysis
        right_panel = QTabWidget()
        
        # Code View Tab
        self.code_view = QsciScintilla()
        self.code_view.setLexer(QsciLexerPython())
        self.code_view.setReadOnly(True)
        right_panel.addTab(self.code_view, "Code View")
        
        # Analysis Tab
        self.analysis_view = QTextEdit()
        self.analysis_view.setReadOnly(True)
        right_panel.addTab(self.analysis_view, "Analysis")
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)

    def update(self):
        """Refresh all dashboard components with latest data"""
        self.update_agent_table()
        self.update_mechanism_list()
        self.update_code_view()
        self.update_analysis_view()

    def update_agent_table(self):
        members = self.execution_engine.current_members
        self.agent_table.setRowCount(len(members))
        
        for row, member in enumerate(members):
            survival = self.execution_engine.compute_survival_chance(member)
            perf_history = self.execution_engine.performance_history.get(member.id, [])
            last_perf = perf_history[-1] if perf_history else 0
            
            code_versions = len(self.execution_engine.code_memory.get(member.id, []))
            
            items = [
                QTableWidgetItem(str(member.id)),
                QTableWidgetItem(f"{member.vitality:.2f}"),
                QTableWidgetItem(f"{member.cargo:.2f}"),
                QTableWidgetItem(f"{survival:.2f}%"),
                QTableWidgetItem(f"{last_perf:.2f}"),
                QTableWidgetItem(str(code_versions))
            ]
            
            for col, item in enumerate(items):
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.agent_table.setItem(row, col, item)

    def update_mechanism_list(self):
        self.mechanism_list.clear()
        current_round = len(self.execution_engine.execution_history['rounds'])
        start_round = max(0, current_round - 5)
        
        for round_data in self.execution_engine.execution_history['rounds'][start_round:]:
            for mod in round_data['mechanism_modifications']['executed']:
                item_text = (
                    f"Round {round_data['round_number']}: Member {mod['member_id']}\n"
                    f"{mod.get('description', 'Mechanism modification')}"
                )
                self.mechanism_list.addItem(item_text)

    def update_code_view(self):
        if not self.agent_table.selectedItems():
            return
            
        selected_row = self.agent_table.currentRow()
        member_id = int(self.agent_table.item(selected_row, 0).text())
        
        # Get latest code file
        code_dir = self.execution_engine.agent_code_path
        code_files = sorted(
            [f for f in os.listdir(code_dir) if f.startswith(f"agent_{member_id}_")],
            key=lambda x: os.path.getmtime(os.path.join(code_dir, x)),
            reverse=True
        )
        
        if code_files:
            latest_file = os.path.join(code_dir, code_files[0])
            with open(latest_file, 'r') as f:
                self.code_view.setText(f.read())

    def update_analysis_view(self):
        if not self.agent_table.selectedItems():
            return
            
        selected_row = self.agent_table.currentRow()
        member_id = int(self.agent_table.item(selected_row, 0).text())
        
        report = self.execution_engine.analysis_reports.get(member_id)
        if report:
            self.analysis_view.setPlainText(
                f"Latest Analysis for Member {member_id}:\n"
                f"{json.dumps(report, indent=2)}"
            )

def launch_dashboard(execution_engine):
    print("Launching dashboard...")
    app = QApplication(sys.argv)
    dashboard = IslandDashboard(execution_engine)
    print("Dashboard created, showing window...")
    dashboard.show()
    print("Starting Qt event loop...")
    return app.exec_() 