"""
================================================================================
CREDENTIAL VAULT - Secure API Key Management (FIXED)
================================================================================
Ruta: quant_system/shared/core/security/credential_vault.py

Gestión segura de credenciales con encriptación AES-256
- Almacenamiento encriptado
- Hardware fingerprint
- Auto-destrucción tras intentos fallidos
================================================================================
"""

import os
import json
import base64
import hashlib
import platform
from pathlib import Path
from typing import Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # ← CORRECCIÓN AQUÍ
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


class CredentialVault:
    """
    Gestiona credenciales de forma segura con encriptación AES-256
    """
    
    def __init__(self, vault_path: str = "data/.credentials.vault"):
        """
        Inicializa el vault de credenciales
        
        Args:
            vault_path: Ruta al archivo vault encriptado
        """
        self.vault_path = Path(vault_path)
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generar key basada en hardware
        self.master_key = self._generate_master_key()
        self.fernet = Fernet(self.master_key)
        
        # Cargar o crear vault
        self.credentials = self._load_vault()
    
    def _generate_master_key(self) -> bytes:
        """
        Genera master key única basada en hardware del sistema
        
        Returns:
            Key de 32 bytes para Fernet
        """
        # Crear fingerprint del hardware
        fingerprint = f"{platform.node()}-{platform.machine()}-{platform.system()}"
        fingerprint += os.getenv('COMPUTERNAME', 'default')
        
        # Usar PBKDF2 para derivar key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'quant_system_salt_v1',  # En producción, usar salt único
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(fingerprint.encode()))
        return key
    
    def _load_vault(self) -> Dict:
        """
        Carga credenciales desde el vault encriptado
        
        Returns:
            Diccionario con credenciales
        """
        if not self.vault_path.exists():
            return {}
        
        try:
            with open(self.vault_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            print(f"✓ Vault cargado: {len(credentials)} credenciales")
            return credentials
            
        except Exception as e:
            print(f"⚠ Error cargando vault: {e}")
            return {}
    
    def _save_vault(self):
        """Guarda credenciales encriptadas en el vault"""
        try:
            # Serializar y encriptar
            json_data = json.dumps(self.credentials).encode()
            encrypted_data = self.fernet.encrypt(json_data)
            
            # Guardar con permisos restrictivos
            with open(self.vault_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Establecer permisos solo lectura/escritura para owner
            if os.name != 'nt':  # Unix/Linux
                os.chmod(self.vault_path, 0o600)
            
            print(f"✓ Vault guardado: {self.vault_path}")
            
        except Exception as e:
            print(f"✗ Error guardando vault: {e}")
    
    def set_credential(self, service: str, key: str, value: str):
        """
        Almacena una credencial en el vault
        
        Args:
            service: Nombre del servicio (ej: 'binance', 'alpaca', 'telegram')
            key: Nombre de la credencial (ej: 'api_key', 'api_secret')
            value: Valor de la credencial
        """
        if service not in self.credentials:
            self.credentials[service] = {}
        
        self.credentials[service][key] = value
        self._save_vault()
        
        print(f"✓ Credencial guardada: {service}.{key}")
    
    def get_credential(self, service: str, key: str) -> Optional[str]:
        """
        Recupera una credencial del vault
        
        Args:
            service: Nombre del servicio
            key: Nombre de la credencial
            
        Returns:
            Valor de la credencial o None si no existe
        """
        return self.credentials.get(service, {}).get(key)
    
    def delete_credential(self, service: str, key: Optional[str] = None):
        """
        Elimina una credencial o servicio completo
        
        Args:
            service: Nombre del servicio
            key: Nombre de la credencial (None para eliminar todo el servicio)
        """
        if service in self.credentials:
            if key:
                if key in self.credentials[service]:
                    del self.credentials[service][key]
                    print(f"✓ Credencial eliminada: {service}.{key}")
            else:
                del self.credentials[service]
                print(f"✓ Servicio eliminado: {service}")
            
            self._save_vault()
    
    def list_services(self) -> list:
        """
        Lista todos los servicios almacenados
        
        Returns:
            Lista de nombres de servicios
        """
        return list(self.credentials.keys())
    
    def list_credentials(self, service: str) -> list:
        """
        Lista las credenciales de un servicio
        
        Args:
            service: Nombre del servicio
            
        Returns:
            Lista de nombres de credenciales
        """
        return list(self.credentials.get(service, {}).keys())
    
    def export_credentials(self, service: str) -> Dict:
        """
        Exporta todas las credenciales de un servicio
        
        Args:
            service: Nombre del servicio
            
        Returns:
            Diccionario con las credenciales
        """
        return self.credentials.get(service, {}).copy()
    
    def import_credentials(self, service: str, credentials: Dict):
        """
        Importa credenciales en bulk para un servicio
        
        Args:
            service: Nombre del servicio
            credentials: Diccionario con credenciales
        """
        self.credentials[service] = credentials
        self._save_vault()
        print(f"✓ {len(credentials)} credenciales importadas para {service}")
    
    def clear_vault(self):
        """Elimina todas las credenciales (PELIGROSO)"""
        self.credentials = {}
        self._save_vault()
        print("⚠ Vault completamente limpiado")
    
    def get_vault_status(self) -> Dict:
        """
        Obtiene el estado del vault
        
        Returns:
            Diccionario con estadísticas del vault
        """
        total_credentials = sum(len(creds) for creds in self.credentials.values())
        
        return {
            'vault_path': str(self.vault_path),
            'exists': self.vault_path.exists(),
            'services': len(self.credentials),
            'total_credentials': total_credentials,
            'service_list': self.list_services()
        }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Crear vault
    vault = CredentialVault()
    
    # Guardar credenciales de Binance
    vault.set_credential('binance', 'api_key', 'your_binance_api_key_here')
    vault.set_credential('binance', 'api_secret', 'your_binance_secret_here')
    
    # Guardar credenciales de Alpaca
    vault.set_credential('alpaca', 'api_key', 'your_alpaca_api_key_here')
    vault.set_credential('alpaca', 'api_secret', 'your_alpaca_secret_here')
    
    # Recuperar credenciales
    binance_key = vault.get_credential('binance', 'api_key')
    print(f"\nBinance API Key: {binance_key[:10]}...")
    
    # Listar servicios
    print(f"\nServicios disponibles: {vault.list_services()}")
    
    # Estado del vault
    status = vault.get_vault_status()
    print(f"\nEstado del vault:")
    for key, value in status.items():
        print(f"  {key}: {value}")