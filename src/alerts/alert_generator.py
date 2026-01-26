import uuid
from datetime import datetime
from typing import Dict, Optional, List
import json


class AlertGenerator:
    """
    Alert generation and notification system
    """
    
    def __init__(self):
        self.alerts = {}  # In-memory alert storage (use database in production)
        
    def generate_alert(
        self,
        transaction_id: str,
        anomaly_score: float,
        risk_level: str,
        details: List[str]
    ) -> str:
        """
        Generate alert for suspicious transaction
        
        Args:
            transaction_id: Transaction identifier
            anomaly_score: Anomaly score
            risk_level: Risk level (LOW/MEDIUM/HIGH/CRITICAL)
            details: Contributing factors
            
        Returns:
            Alert ID
        """
        alert_id = f"alert_{uuid.uuid4().hex[:12]}"
        
        alert = {
            'alert_id': alert_id,
            'transaction_id': transaction_id,
            'anomaly_score': anomaly_score,
            'risk_level': risk_level,
            'details': details,
            'status': 'OPEN',
            'created_at': datetime.now().isoformat(),
            'assigned_to': None,
            'resolution': None
        }
        
        self.alerts[alert_id] = alert
        
        return alert_id
    
    def get_alert(self, alert_id: str) -> Optional[Dict]:
        """Retrieve alert by ID"""
        return self.alerts.get(alert_id)
    
    def get_alerts_by_status(self, status: str) -> List[Dict]:
        """Get all alerts with given status"""
        return [
            alert for alert in self.alerts.values()
            if alert['status'] == status
        ]
    
    def update_alert(
        self, 
        alert_id: str, 
        updates: Dict
    ) -> bool:
        """Update alert details"""
        if alert_id in self.alerts:
            self.alerts[alert_id].update(updates)
            self.alerts[alert_id]['updated_at'] = datetime.now().isoformat()
            return True
        return False
    
    def send_notification(self, alert_id: str):
        """
        Send notification for alert
        (Implement email/Slack integration here)
        """
        alert = self.get_alert(alert_id)
        if not alert:
            return
        
        # Log notification (implement actual sending)
        print(f"📧 ALERT NOTIFICATION: {alert_id}")
        print(f"Risk Level: {alert['risk_level']}")
        print(f"Score: {alert['anomaly_score']:.3f}")
        print(f"Details: {', '.join(alert['details'])}")
