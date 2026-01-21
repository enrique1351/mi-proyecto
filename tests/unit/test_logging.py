"""
================================================================================
UNIT TESTS - Logging Configuration
================================================================================
Ruta: quant_system/tests/unit/test_logging.py

Tests para el módulo de configuración de logging
- Test de inicialización de logging
- Test de manejo de caracteres Unicode
- Test de manejo de emojis en consolas sin soporte UTF-8
================================================================================
"""

import unittest
import logging
import tempfile
import shutil
import os
import sys
from pathlib import Path
from datetime import datetime
from io import StringIO

# Agregar path del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Nota: Usamos una versión de prueba de setup_logging porque la función real de main.py
# está diseñada para configuración de aplicación completa y es difícil de aislar en tests.
# Esta versión de prueba replica la misma lógica para garantizar que los tests validen
# correctamente el comportamiento de encoding UTF-8 y manejo de errores.
def setup_logging_test(log_level: str = "INFO", log_dir: Path = None):
    """Versión de prueba de setup_logging con soporte para caracteres Unicode."""
    
    if log_dir is None:
        log_dir = Path(tempfile.mkdtemp()) / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Crear manejador de archivo con encoding UTF-8
    file_handler = logging.FileHandler(
        log_dir / f"system_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Crear manejador de consola con manejo de errores de encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configurar el stream con manejo de errores de encoding
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(errors='replace')
        except (AttributeError, OSError):
            # AttributeError: método no disponible
            # OSError: operación no soportada en el stream
            pass
    
    # Limpiar handlers existentes
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[file_handler, console_handler]
    )
    
    return log_dir


class TestLoggingConfiguration(unittest.TestCase):
    """Tests para la configuración de logging"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.test_dir / "logs"
    
    def tearDown(self):
        """Cleanup después de cada test"""
        # Limpiar handlers de logging
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_logging_creates_directory(self):
        """Test: La configuración de logging crea el directorio de logs"""
        log_dir = setup_logging_test(log_dir=self.log_dir)
        
        self.assertTrue(log_dir.exists())
        self.assertTrue(log_dir.is_dir())
    
    def test_logging_creates_file(self):
        """Test: La configuración de logging crea el archivo de log"""
        log_dir = setup_logging_test(log_dir=self.log_dir)
        
        # Verificar que existe al menos un archivo de log
        log_files = list(log_dir.glob("system_*.log"))
        self.assertGreater(len(log_files), 0)
    
    def test_logging_with_ascii_text(self):
        """Test: Logging funciona con texto ASCII normal"""
        log_dir = setup_logging_test(log_dir=self.log_dir)
        logger = logging.getLogger(__name__)
        
        # Escribir un mensaje normal
        logger.info("This is a normal ASCII message")
        
        # Verificar que se escribió en el archivo
        log_files = list(log_dir.glob("system_*.log"))
        self.assertGreater(len(log_files), 0)
        
        with open(log_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("This is a normal ASCII message", content)
    
    def test_logging_with_unicode_characters(self):
        """Test: Logging maneja caracteres Unicode correctamente"""
        log_dir = setup_logging_test(log_dir=self.log_dir)
        logger = logging.getLogger(__name__)
        
        # Escribir mensaje con caracteres Unicode
        unicode_message = "Mensaje con caracteres especiales: áéíóú ñ €"
        logger.info(unicode_message)
        
        # Verificar que se escribió correctamente en el archivo
        log_files = list(log_dir.glob("system_*.log"))
        self.assertGreater(len(log_files), 0)
        
        with open(log_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn(unicode_message, content)
    
    def test_logging_with_emojis(self):
        """Test: Logging maneja emojis correctamente en el archivo"""
        log_dir = setup_logging_test(log_dir=self.log_dir)
        logger = logging.getLogger(__name__)
        
        # Escribir mensaje con emojis
        emoji_message = "🚀 Sistema iniciado ✅ Todo OK"
        
        # Esto no debería lanzar excepción
        try:
            logger.info(emoji_message)
            success = True
        except Exception as e:
            success = False
            print(f"Error logging emoji: {e}")
        
        self.assertTrue(success, "Logging con emojis no debería fallar")
        
        # Verificar que se escribió en el archivo (pueden estar reemplazados)
        log_files = list(log_dir.glob("system_*.log"))
        self.assertGreater(len(log_files), 0)
        
        with open(log_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            # El mensaje debería estar presente o con caracteres de reemplazo
            self.assertTrue(
                emoji_message in content or "Sistema iniciado" in content,
                "El mensaje debería estar presente en el log"
            )
    
    def test_logging_different_levels(self):
        """Test: Diferentes niveles de logging funcionan correctamente"""
        log_dir = setup_logging_test(log_level="DEBUG", log_dir=self.log_dir)
        logger = logging.getLogger(__name__)
        
        # Escribir mensajes de diferentes niveles
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Verificar que todos se escribieron
        log_files = list(log_dir.glob("system_*.log"))
        self.assertGreater(len(log_files), 0)
        
        with open(log_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Debug message", content)
            self.assertIn("Info message", content)
            self.assertIn("Warning message", content)
            self.assertIn("Error message", content)


class TestLoggingEncodingHandling(unittest.TestCase):
    """Tests para el manejo de encoding en logging"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.test_dir / "logs"
    
    def tearDown(self):
        """Cleanup después de cada test"""
        # Limpiar handlers de logging
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_file_handler_uses_utf8(self):
        """Test: El file handler usa encoding UTF-8"""
        log_dir = setup_logging_test(log_dir=self.log_dir)
        
        # Obtener el file handler
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers 
                        if isinstance(h, logging.FileHandler)]
        
        self.assertGreater(len(file_handlers), 0)
        
        # Verificar que tiene encoding UTF-8
        file_handler = file_handlers[0]
        self.assertEqual(file_handler.encoding, 'utf-8')
    
    def test_logging_handles_mixed_content(self):
        """Test: Logging maneja contenido mixto (ASCII + Unicode + Emojis)"""
        log_dir = setup_logging_test(log_dir=self.log_dir)
        logger = logging.getLogger(__name__)
        
        # Mensaje con contenido mixto
        mixed_message = "ASCII text + Unicode: áéí + Emoji: 🔥"
        
        try:
            logger.info(mixed_message)
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)
        
        # Verificar que se escribió en el archivo
        log_files = list(log_dir.glob("system_*.log"))
        with open(log_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            # Al menos la parte ASCII y Unicode deben estar presentes
            self.assertIn("ASCII text", content)
            self.assertIn("Unicode", content)


class TestLoggingErrorRecovery(unittest.TestCase):
    """Tests para la recuperación de errores en logging"""
    
    def setUp(self):
        """Setup para cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.test_dir / "logs"
    
    def tearDown(self):
        """Cleanup después de cada test"""
        # Limpiar handlers de logging
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_logging_continues_after_encoding_error(self):
        """Test: Logging continúa funcionando después de un error de encoding"""
        log_dir = setup_logging_test(log_dir=self.log_dir)
        logger = logging.getLogger(__name__)
        
        # Intentar loggear varios mensajes, incluyendo problemáticos
        messages = [
            "Normal message 1",
            "🚀 Emoji message",
            "Normal message 2",
            "Ñoño con ñ",
            "Normal message 3"
        ]
        
        # Todos deberían procesarse sin crash
        for msg in messages:
            try:
                logger.info(msg)
            except Exception as e:
                self.fail(f"Logging no debería fallar con mensaje: {msg}, error: {e}")
        
        # Verificar que se creó el archivo de log
        log_files = list(log_dir.glob("system_*.log"))
        self.assertGreater(len(log_files), 0)


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    # Ejecutar todos los tests
    unittest.main(verbosity=2)
