import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
import json
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

# Modelo LLM para análisis médico
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Definiciones de modelos de datos
class TipoEvento(str, Enum):
    CENTINELA = "Evento Centinela"
    ADVERSO = "Evento Adverso"
    INCIDENTE = "Incidente"
    SEGURIDAD = "Evento Relacionado con la Seguridad del Paciente"
    DAÑO = "Daño"
    ERROR_SISTEMA = "Error de Sistema"
    NINGUNO = "Ninguno"

class NivelRiesgo(str, Enum):
    ALTO = "Alto Riesgo"
    MODERADO = "Riesgo Moderado"
    BAJO = "Bajo Riesgo"

class EventoAdverso(TypedDict):
    """Estructura de un evento adverso detectado"""
    tipo: str  # Tipo de evento
    descripcion: str  # Descripción del evento

class Estado(TypedDict):
    """Estado del análisis de revisión médica con historial en español argentino"""
    revision_actual: str  # Revisión médica actual que se está analizando
    revisiones_historicas: Annotated[List[str], operator.add] = Field(default_factory=list)  # Historial de todas las revisiones
    resumen_evolucion: Optional[str] = Field(default="")  # Resumen de la evolución del paciente
    eventos_adversos: Annotated[List[EventoAdverso], operator.add] = Field(default_factory=list)  # Lista de eventos adversos detectados
    nivel_riesgo: str = Field(default=f"{NivelRiesgo.BAJO.value} - Evaluación inicial")  # Nivel de riesgo general del paciente
    resultado_formateado: Optional[str] = Field(default="")  # Resultado formateado para mostrar

# Modelos para structured output
class DeteccionEventoCompleto(BaseModel):
    evento_detectado: bool = Field(description="Indica si se detectó algún evento adverso o de seguridad")
    tipo_evento: Optional[TipoEvento] = Field(description="Tipo de evento detectado", default=None)
    descripcion: Optional[str] = Field(description="Descripción del evento detectado", default=None)
    urgencia: Optional[int] = Field(description="Nivel de urgencia del evento (1-5)", default=None)

class ResumenEvolucion(BaseModel):
    resumen: str = Field(description="Resumen actualizado de la evolución del paciente")
    recomendaciones: List[str] = Field(description="Recomendaciones principales")

# Definiciones de los tipos de eventos
DEFINICIONES_EVENTOS = """
TIPOS DE EVENTOS:

- Evento Centinela: Un evento centinela es un suceso inesperado relacionado con la atención médica que resulta en muerte, daño físico o psicológico grave, o riesgo de que ocurran. Estos eventos señalan fallas serias en los sistemas de atención médica que requieren una investigación y acción inmediata. Se puede considerar un evento centinela a un evento que si bien no dejó secuelas puso en una grave situación al paciente (por ejemplo paro cardiorrespiratorio con restitución completa). 
Ejemplos:
• Cirugía en el sitio equivocado.
• Muerte materna no relacionada con las condiciones subyacentes.
• Reacción hemolítica por incompatibilidad de grupo sanguíneo.
• Suicidio de un paciente hospitalizado.

- Evento Adverso: Un evento adverso es un daño no intencionado que ocurre durante la atención sanitaria y que no está relacionado con la enfermedad subyacente del paciente.
Ejemplos:
• Infecciones asociadas a dispositivos médicos.
• Daño por errores de medicación (como sobredosis).
• Caídas del paciente durante la estancia hospitalaria.

- Incidente: Un incidente es un evento o circunstancia que podría haber causado daño al paciente, pero no lo hizo, ya sea porque fue detectado a tiempo o por puro azar.
Subtipos:
• Cuasi-incidente (Near Miss): Un error que podría haber causado daño, pero no lo hizo gracias a una intervención o circunstancia fortuita.
• Condición insegura: Una situación que incrementa el riesgo de que ocurra un incidente, como equipos médicos defectuosos.

- Evento Relacionado con la Seguridad del Paciente: Un evento que no necesariamente implica daño, pero está relacionado con riesgos o fallas en los procesos de atención.
Ejemplo:
• Retraso en la administración de un tratamiento crítico.

- Daño: El daño es el impacto negativo para el paciente, resultado de la atención médica, que puede incluir:
• Daño físico (lesiones, infecciones).
• Daño psicológico (estrés, trauma).
• Incremento en los días de hospitalización.

- Error de Sistema: Un error de sistema se refiere a fallas en los procesos, procedimientos o equipos que contribuyen a los incidentes o eventos adversos.

NIVELES DE URGENCIA:
1 - Urgencia Inmediata (Requiere acción inmediata, riesgo vital)
2 - Urgencia Alta (Requiere acción dentro de horas)
3 - Urgencia Media (Requiere acción dentro de un día)
4 - Urgencia Baja (Requiere seguimiento)
5 - Sin Urgencia (Solo registro)

REGLA DE CLASIFICACIÓN DE RIESGO:
La presencia de estos puntos se clasificará como internación de riesgo. De presentar más de uno o un evento Centinela de muy alto riesgo.
"""

def detectar_y_clasificar_evento(state):
    """Detecta y clasifica eventos adversos en la revisión médica actual"""
    revision = state["revision_actual"]
    
    # Crear un parser para la salida estructurada
    parser = JsonOutputParser(pydantic_object=DeteccionEventoCompleto)
    
    prompt = f"""Analizá la siguiente revisión médica para detectar y clasificar si hay algún evento adverso:
    
{DEFINICIONES_EVENTOS}

REVISIÓN MÉDICA:
{revision}

Si detectás algún evento, clasificalo según su tipo e incluí una descripción detallada.
Si no detectás ningún evento, respondé con evento_detectado: false.

{parser.get_format_instructions()}
"""
    
    # Obtener respuesta estructurada
    respuesta_chain = llm.with_structured_output(DeteccionEventoCompleto)
    resultado = respuesta_chain.invoke(prompt)
    
    # Si no se detectó ningún evento, retornar sin cambios
    if not resultado.evento_detectado:
        return {
            "revisiones_historicas": state["revisiones_historicas"] + [state["revision_actual"]]
        }
    
    # Crear el evento adverso
    nuevo_evento = EventoAdverso(
        tipo=resultado.tipo_evento,
        descripcion=resultado.descripcion or "Sin descripción disponible"
    )
    
    # Agregar a la lista de eventos adversos
    return {
        "eventos_adversos": [nuevo_evento],
        "revisiones_historicas": [state["revision_actual"]]
    }

def determinar_nivel_riesgo(state):
    """Determina el nivel de riesgo basado en los eventos adversos"""
    eventos = state["eventos_adversos"]
    
    # Verificar la presencia de eventos centinela de alto riesgo (urgencia 1-2)
    eventos_centinela = [e for e in eventos if e["tipo"] == TipoEvento.CENTINELA]
    
    # Aplicar la regla: "La presencia de estos puntos se clasificará como internación de riesgo.
    # De presentar más de uno o un evento Centinela de muy alto riesgo."
    total_eventos = len(eventos)
    
    if len(eventos_centinela) > 0:
        nivel_riesgo = f"{NivelRiesgo.ALTO.value} - Evento Centinela detectado"
    elif total_eventos > 1:
        nivel_riesgo = f"{NivelRiesgo.ALTO.value} - Múltiples eventos detectados ({total_eventos})"
    elif total_eventos == 1:
        nivel_riesgo = f"{NivelRiesgo.MODERADO.value} - Un evento detectado"
    else:
        nivel_riesgo = f"{NivelRiesgo.BAJO.value} - Sin eventos detectados"
    
    return {
        "nivel_riesgo": nivel_riesgo
    }

def generar_resumen_evolucion(state):
    """Genera un resumen actualizado de la evolución del paciente"""
    revision_actual = state["revision_actual"]
    revisiones_historicas = state.get("revisiones_historicas", [])
    eventos_adversos = state.get("eventos_adversos", [])
    nivel_riesgo = state.get("nivel_riesgo", f"{NivelRiesgo.BAJO.value} - Sin clasificación")
    resumen_previo = state.get("resumen_evolucion", "")
    
    # Información de eventos para el resumen
    info_eventos = []
    for idx, evento in enumerate(eventos_adversos):
        info_eventos.append(f"Evento {idx+1}: {evento['tipo']} - {evento['descripcion'][:100]}...")
    
    # Si es la primera revisión, simplificar
    if len(revisiones_historicas) <= 1:
        # Preparar el texto de eventos detectados
        texto_eventos = "No se detectaron eventos adversos."
        if info_eventos:
            texto_eventos = "EVENTOS DETECTADOS:\n" + json.dumps(info_eventos, ensure_ascii=False, indent=2)
            
        prompt = f"""Generá un resumen inicial en español argentino (usando "vos" en lugar de "tú") para la revisión médica:
        
REVISIÓN MÉDICA:
{revision_actual}

{texto_eventos}

NIVEL DE RIESGO:
{nivel_riesgo}

El resumen debe ser claro, conciso y profesional, escrito en español argentino.
"""
        respuesta = llm.invoke([SystemMessage(content=prompt)])
        
        return {
            "resumen_evolucion": respuesta.content
        }
    
    # Para revisiones posteriores, incorporar la historia
    parser = JsonOutputParser(pydantic_object=ResumenEvolucion)
    
    prompt = f"""Generá un resumen actualizado de la evolución del paciente en español argentino (usando "vos" en lugar de "tú").
    
RESUMEN PREVIO:
{resumen_previo or "No disponible"}

REVISIÓN ACTUAL:
{revision_actual}

REVISIONES HISTÓRICAS (TOTAL: {len(revisiones_historicas)}):
{revisiones_historicas[-1] if len(revisiones_historicas) > 0 else "No hay revisiones previas"}

EVENTOS ADVERSOS ({len(eventos_adversos)}):
{json.dumps(info_eventos, ensure_ascii=False, indent=2) if info_eventos else "No se detectaron eventos adversos."}

NIVEL DE RIESGO:
{nivel_riesgo}

Tu resumen debe:
1. Integrar la información nueva con la previa
2. Destacar cambios importantes en la condición del paciente
3. Mencionar todos los eventos adversos relevantes
4. Indicar el nivel de riesgo actual
5. Proveer recomendaciones basadas en la situación

{parser.get_format_instructions()}
"""
    
    respuesta_chain = llm.with_structured_output(ResumenEvolucion)
    resultado = respuesta_chain.invoke(prompt)
    
    return {
        "resumen_evolucion": resultado.resumen
    }

def formatear_resultado(state):
    """Genera un string formateado con toda la información relevante"""
    revision_actual = state["revision_actual"]
    revisiones_historicas = state.get("revisiones_historicas", [])
    resumen_evolucion = state.get("resumen_evolucion", "No hay resumen disponible")
    eventos_adversos = state.get("eventos_adversos", [])
    nivel_riesgo = state.get("nivel_riesgo", f"{NivelRiesgo.BAJO.value} - Sin clasificación")
    
    # Contar eventos por tipo
    conteo_por_tipo = {}
    for evento in eventos_adversos:
        tipo = evento["tipo"]
        conteo_por_tipo[tipo] = conteo_por_tipo.get(tipo, 0) + 1
    
    # Formatear el resultado
    resultado = f"""
=== INFORME DE SEGUIMIENTO MÉDICO ===

NIVEL DE RIESGO: {nivel_riesgo}

RESUMEN DE EVOLUCIÓN:
{resumen_evolucion}

EVENTOS ADVERSOS DETECTADOS ({len(eventos_adversos)}):
"""
    
    if eventos_adversos:
        for i, evento in enumerate(eventos_adversos):
            resultado += f"\n{i+1}. {evento['tipo']} - {evento['descripcion'][:150]}..."
    else:
        resultado += "\nNo se han detectado eventos adversos hasta el momento."
        
    resultado += f"""

ESTADÍSTICAS:
- Cantidad de revisiones: {len(revisiones_historicas)}
- Distribución de eventos: {json.dumps(conteo_por_tipo, ensure_ascii=False)}

===================================
"""
    
    return {
        "resultado_formateado": resultado
    }

# Creamos el grafo de estados
builder = StateGraph(Estado)

# Agregamos los nodos
builder.add_node("detectar_y_clasificar_evento", detectar_y_clasificar_evento)
builder.add_node("determinar_nivel_riesgo", determinar_nivel_riesgo)
builder.add_node("generar_resumen_evolucion", generar_resumen_evolucion)
builder.add_node("formatear_resultado", formatear_resultado)

# Definimos el flujo
builder.add_edge(START, "detectar_y_clasificar_evento")
builder.add_edge("detectar_y_clasificar_evento", "determinar_nivel_riesgo")
builder.add_edge("determinar_nivel_riesgo", "generar_resumen_evolucion")
builder.add_edge("generar_resumen_evolucion", "formatear_resultado")
builder.add_edge("formatear_resultado", END)

# Compilamos el grafo
graph = builder.compile()

# Función para procesar una nueva revisión médica
def procesar_nueva_revision(estado_actual, nueva_revision):
    """
    Procesa una nueva revisión médica y actualiza el estado del paciente
    
    Args:
        estado_actual (dict): Estado actual del paciente (None para el primer caso)
        nueva_revision (str): Texto de la nueva revisión médica
        
    Returns:
        dict: Estado actualizado del paciente con el resultado formateado
    """
    if estado_actual is None:
        # Primer caso, inicializar estado
        estado_inicial = {
            "revision_actual": nueva_revision,
            "revisiones_historicas": [],
            "resumen_evolucion": "",
            "eventos_adversos": [],
            "nivel_riesgo": f"{NivelRiesgo.BAJO.value} - Evaluación inicial",
            "resultado_formateado": "",
        }
    else:
        # Continuar con un caso existente
        estado_inicial = {
            "revision_actual": nueva_revision,
            "revisiones_historicas": estado_actual.get("revisiones_historicas", []),
            "resumen_evolucion": estado_actual.get("resumen_evolucion", ""),
            "eventos_adversos": estado_actual.get("eventos_adversos", []),
            "nivel_riesgo": estado_actual.get("nivel_riesgo", f"{NivelRiesgo.BAJO.value} - Sin clasificación"),
            "resultado_formateado": estado_actual.get("resultado_formateado", ""),
        }
    
    # Ejecutar el grafo
    resultado = graph.invoke(estado_inicial)
    return resultado
