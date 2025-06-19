extends Node3D

class_name Hand

const NUM_LANDMARKS: int = 21
const HAND_SCALE: float = 30.0

const HAND_LINES_MAPPING = [
	[0, 1], [1, 2], [2, 3], [3, 4], # Thumb
	[0, 5], [5, 6], [6, 7], [7, 8], # Index Finger
	[5, 9], [9, 10], [10, 11], [11, 12], # Middle Finger
	[9, 13], [13, 14], [14, 15], [15, 16], # Ring Finger
	[0, 17], [13, 17], [17, 18], [18, 19], [19, 20], # Pinky
]

var landmark_sphere: PackedScene = preload("res://hand/hand_landmark.tscn")

# adiciona uma cena para o modelo de mão rigged
@export var rigged_hand_model_path: String = "res://../models/hand.glb" # caminho para o seu modelo de mão
var rigged_hand_scene: PackedScene
var rigged_hand_instance: Node3D
var hand_skeleton: Skeleton3D

var hand_landmarks: Array[HandLandmark] = []
var hand_lines: Array[MeshInstance3D] = [] # para o modo wireframe
var hand_solid_mesh: Node3D # Alterado para Node3D para ser um contentor puro
var mesh_material: StandardMaterial3D
var show_wireframe: bool = false
var show_solid: bool = true

# Arrays para guardar as MeshInstance3D das juntas e ossos
var joint_visuals: Array[MeshInstance3D] = []
var bone_visuals: Array[MeshInstance3D] = []

# Parâmetros de qualidade (ajustados para um aspeto mais preenchido)
var joint_radius_base = 0.5 # Aumentado de 0.3
var joint_radius_finger = 0.4 # Aumentado de 0.25
var joint_radius_tip = 0.35 # Aumentado de 0.2
var bone_capsule_radius = 0.3 # Aumentado de 0.15
var segments_high = 16 # Aumentado de 12
var segments_medium = 10 # Aumentado de 8


func _ready() -> void:
	rigged_hand_scene = load(rigged_hand_model_path) # carrega a cena do modelo de mão
	_create_hand_landmark_spheres() # Isto cria os HandLandmark (Node3D com script)
	_create_hand_lines_visuals() # Para o modo wireframe
	_create_solid_hand_visuals() # agora vai instanciar o modelo rigged

func _create_hand_landmark_spheres() -> void:
	for i in range(NUM_LANDMARKS):
		var landmark_instance = landmark_sphere.instantiate() as HandLandmark
		landmark_instance.from_landmark_id(i)
		add_child(landmark_instance)
		hand_landmarks.append(landmark_instance)
		landmark_instance.visible = false # Escondidos por padrão, são apenas pontos de dados

func _create_hand_lines_visuals() -> void: # Renomeado para clareza
	for i in range(HAND_LINES_MAPPING.size()):
		var line_instance := MeshInstance3D.new()
		line_instance.visible = false
		add_child(line_instance) # Adiciona diretamente ao nó Hand, não ao hand_solid_mesh
		hand_lines.append(line_instance)

func _create_solid_hand_visuals() -> void:
	# remove a criação de malhas sólidas primitivas (esferas e cápsulas)
	# em vez disso, instancia o modelo de mão rigged
	if rigged_hand_scene:
		rigged_hand_instance = rigged_hand_scene.instantiate() as Node3D
		if rigged_hand_instance:
			add_child(rigged_hand_instance)
			# tenta encontrar o nó Skeleton3D dentro da instância da mão
			# o nome 'Skeleton3D' pode variar dependendo de como o modelo foi exportado
			hand_skeleton = rigged_hand_instance.find_child("Skeleton3D", true, false) 
			if not hand_skeleton:
				# pode ser necessário um nome diferente ou um caminho direto se 'Skeleton3D' não for encontrado
				# por exemplo, se o Skeleton3D for o próprio nó raiz do GLTF, ou tiver outro nome.
				# Se o nó raiz do GLTF é o Skeleton3D:
				# if rigged_hand_instance is Skeleton3D:
				# hand_skeleton = rigged_hand_instance as Skeleton3D
				printerr("Skeleton3D node not found in the rigged hand model. Please check the model structure and node name.")
			rigged_hand_instance.visible = show_solid
		else:
			printerr("Failed to instantiate rigged hand model.")
	else:
		printerr("Rigged hand model scene not loaded. Check path: ", rigged_hand_model_path)

	# o material antigo e as visuals de joints/bones não são mais necessários aqui
	# mesh_material = StandardMaterial3D.new() 
	# ... (resto do código de criação de material e joint_visuals/bone_visuals removido)


func _process(_delta: float) -> void:
	if show_wireframe:
		_update_hand_lines_visuals()
	
	# if show_solid and hand_solid_mesh.visible: # Apenas atualiza se estiver visível
	# 	_update_hand_solid_mesh()
	# substitui a linha acima pela verificação do novo modelo de mão
	if show_solid and rigged_hand_instance and rigged_hand_instance.visible and hand_skeleton:
		_update_rigged_hand_pose() # nova função para manipular os ossos

func _update_hand_lines_visuals() -> void: # Renomeado para clareza
	for i in range(HAND_LINES_MAPPING.size()):
		var mapping = HAND_LINES_MAPPING[i]
		var p0 = hand_landmarks[mapping[0]].global_position
		var p1 = hand_landmarks[mapping[1]].global_position
		LineRenderer.edit_line(hand_lines[i], p0, p1)


# func _update_hand_solid_mesh() -> void: # esta função será substituída
# 	# ... (todo o conteúdo antigo desta função que movia esferas e cápsulas é removido)

# nova função para atualizar a pose do esqueleto da mão rigged
func _update_rigged_hand_pose() -> void:
	if not hand_skeleton or hand_landmarks.size() != NUM_LANDMARKS:
		return

	# obtém as posições globais dos landmarks.
	# hand_landmarks[i].global_position já deve estar em coordenadas globais
	# devido à forma como HandLandmark.gd atualiza sua própria global_position.
	var lm_global_pos: Array[Vector3] = []
	for i in range(NUM_LANDMARKS):
		lm_global_pos.append(hand_landmarks[i].global_position)

	# --- Mapeamento de Nomes de Ossos (VERIFIQUE E AJUSTE ESTES NOMES!) ---
	# Estes nomes são baseados na sua captura de tela.
	# Confirme se eles são os nomes corretos que o hand_skeleton.find_bone() espera.
	# Você pode ver os nomes exatos dos ossos selecionando o Skeleton3D no editor Godot.
	var BONE_NAMES = {
		"ARM_LOWER": "arm_lower", # Osso principal do pulso/braço
		"THUMB_0": "thumb_0", "THUMB_1": "thumb_1", "THUMB_2": "thumb_2", # thumb_2 pode ter outro nome ou não existir como nó visível
		"INDEX_0": "index_0", "INDEX_1": "index_1", "INDEX_2": "index_2",
		"MIDDLE_0": "middle_0", "MIDDLE_1": "middle_1", "MIDDLE_2": "middle_2",
		"RING_0": "ring_0", "RING_1": "ring_1", "RING_2": "ring_2",
		"PINKY_0": "pinky_0", "PINKY_1": "pinky_1", "PINKY_2": "pinky_2"
	}
	# Landmarks do MediaPipe:
	# WRIST = 0
	# THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4
	# INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8
	# MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12
	# RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16
	# PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20

	# --- Cálculo do Vetor "Up" de Referência da Mão (Global) ---
	# Este é um ponto crucial e difícil de acertar genericamente.
	# Vamos tentar uma aproximação:
	# Vetor "forward" da mão (do pulso para o meio da base dos dedos)
	var hand_root_pos = lm_global_pos[0] # WRIST
	var middle_mcp_pos = lm_global_pos[9] # MIDDLE_FINGER_MCP
	var hand_forward_global = (middle_mcp_pos - hand_root_pos).normalized()

	# Vetor "lateral" da mão (do centro da palma para o polegar, por exemplo)
	# Ou, melhor, um vetor que define o plano da palma.
	# Da base do dedo indicador (LM5) para a base do mindinho (LM17) pode dar uma ideia da largura da palma.
	var index_mcp_pos = lm_global_pos[5]
	var pinky_mcp_pos = lm_global_pos[17]
	var hand_side_vector_palm = (pinky_mcp_pos - index_mcp_pos).normalized()
	
	# O normal da palma pode ser o produto vetorial de 'forward' e 'side'
	# A ordem importa para a direção (para fora ou para dentro da palma)
	var palm_normal_global = hand_forward_global.cross(hand_side_vector_palm).normalized()
	
	# O vetor "up" da mão (perpendicular ao 'forward' e ao 'palm_normal', ou seja, ao longo do 'side_vector_palm' ajustado)
	# Este seria o vetor que aponta para "cima" se a palma estiver virada para a frente.
	var hand_up_vector_global = palm_normal_global.cross(hand_forward_global).normalized()
	if hand_up_vector_global.length_squared() < 0.5: # Fallback
		hand_up_vector_global = Vector3.UP # Fallback muito genérico

	# --- Orientar o Osso Principal (ARM_LOWER) ---
	var arm_lower_bone_idx = hand_skeleton.find_bone(BONE_NAMES.ARM_LOWER)
	if arm_lower_bone_idx != -1:
		# A posição do osso principal é a origem do esqueleto (local ao Skeleton3D node).
		# Sua rotação definirá a orientação da mão inteira.
		var arm_lower_rest_transform = hand_skeleton.get_bone_rest(arm_lower_bone_idx)
		
		# Queremos que o osso 'arm_lower' (que representa o pulso/antebraço)
		# aponte na direção 'hand_forward_global', com 'hand_up_vector_global' como seu "up".
		var arm_lower_new_local_basis = Basis().looking_at(hand_forward_global, hand_up_vector_global)
		
		# Como arm_lower é provavelmente um osso raiz (ou filho direto do Skeleton3D),
		# sua transformação local é relativa ao próprio Skeleton3D.
		# Se o Skeleton3D estiver alinhado com o mundo, então local_basis = global_basis.
		# Precisamos transformar a base global desejada para a base local do osso.
		var skel_global_inv_basis = hand_skeleton.global_transform.basis.orthonormalized().transposed()
		arm_lower_new_local_basis = skel_global_inv_basis * arm_lower_new_local_basis

		hand_skeleton.set_bone_pose(arm_lower_bone_idx, Transform3D(arm_lower_new_local_basis.orthonormalized(), arm_lower_rest_transform.origin))


	# --- Orientar Ossos dos Dedos ---
	# Argumentos para _orient_bone_locally: nome_osso, lm_inicio_osso_global, lm_fim_osso_global, up_vector_referencia_global
	# O up_vector_referencia_global pode ser o palm_normal_global ou o hand_up_vector_global, dependendo da flexão.
	# Para flexão/extensão dos dedos, o eixo de rotação é geralmente perpendicular à palma.
	# Vamos usar palm_normal_global como o "up" para look_at, o que significa que o "lado" do osso vai alinhar com palm_normal_global.

	# Polegar (Thumb) - Landmarks 1(CMC), 2(MCP), 3(IP), 4(TIP)
	# O polegar é mais complexo, seu 'up' vector pode ser diferente.
	var thumb_up_vector = hand_forward_global # Tentativa para o polegar, pode precisar de ajuste.
	_orient_bone_locally(BONE_NAMES.THUMB_0, lm_global_pos[1], lm_global_pos[2], thumb_up_vector)
	_orient_bone_locally(BONE_NAMES.THUMB_1, lm_global_pos[2], lm_global_pos[3], thumb_up_vector)
	if BONE_NAMES.THUMB_2: _orient_bone_locally(BONE_NAMES.THUMB_2, lm_global_pos[3], lm_global_pos[4], thumb_up_vector)

	# Indicador (Index) - Landmarks 5(MCP), 6(PIP), 7(DIP), 8(TIP)
	_orient_bone_locally(BONE_NAMES.INDEX_0, lm_global_pos[5], lm_global_pos[6], palm_normal_global)
	_orient_bone_locally(BONE_NAMES.INDEX_1, lm_global_pos[6], lm_global_pos[7], palm_normal_global)
	_orient_bone_locally(BONE_NAMES.INDEX_2, lm_global_pos[7], lm_global_pos[8], palm_normal_global)

	# Médio (Middle) - Landmarks 9(MCP), 10(PIP), 11(DIP), 12(TIP)
	_orient_bone_locally(BONE_NAMES.MIDDLE_0, lm_global_pos[9], lm_global_pos[10], palm_normal_global)
	_orient_bone_locally(BONE_NAMES.MIDDLE_1, lm_global_pos[10], lm_global_pos[11], palm_normal_global)
	_orient_bone_locally(BONE_NAMES.MIDDLE_2, lm_global_pos[11], lm_global_pos[12], palm_normal_global)

	# Anelar (Ring) - Landmarks 13(MCP), 14(PIP), 15(DIP), 16(TIP)
	_orient_bone_locally(BONE_NAMES.RING_0, lm_global_pos[13], lm_global_pos[14], palm_normal_global)
	_orient_bone_locally(BONE_NAMES.RING_1, lm_global_pos[14], lm_global_pos[15], palm_normal_global)
	_orient_bone_locally(BONE_NAMES.RING_2, lm_global_pos[15], lm_global_pos[16], palm_normal_global)

	# Mínimo (Pinky) - Landmarks 17(MCP), 18(PIP), 19(DIP), 20(TIP)
	_orient_bone_locally(BONE_NAMES.PINKY_0, lm_global_pos[17], lm_global_pos[18], palm_normal_global)
	_orient_bone_locally(BONE_NAMES.PINKY_1, lm_global_pos[18], lm_global_pos[19], palm_normal_global)
	_orient_bone_locally(BONE_NAMES.PINKY_2, lm_global_pos[19], lm_global_pos[20], palm_normal_global)

	# Forçar o esqueleto a atualizar suas transformações.
	hand_skeleton.force_update_all_bone_transforms()

# --- Função Auxiliar para Orientar Ossos ---
# Orienta um osso para que ele "aponte" de current_bone_pivot_global para next_joint_global_pos_target
# current_bone_pivot_global é a posição global da junta que este osso rotaciona.
# next_joint_global_pos_target é para onde o "fim" do osso deve apontar.
# reference_up_vector_global é o vetor "para cima" desejado no espaço global.
func _orient_bone_locally(bone_name: String, current_bone_pivot_global: Vector3, next_joint_global_pos_target: Vector3, reference_up_vector_global: Vector3) -> void:
	var bone_idx = hand_skeleton.find_bone(bone_name)
	if bone_idx == -1:
		#printerr("Osso não encontrado em _orient_bone_locally: ", bone_name)
		return

	var direction_global = (next_joint_global_pos_target - current_bone_pivot_global).normalized()
	if direction_global.length_squared() < 0.0001: # Evitar direção zero
		#printerr("Direção zero para o osso: ", bone_name)
		return

	# Obter a transformação global do osso pai (ou do Skeleton3D se for um osso raiz relativo ao esqueleto)
	var parent_global_transform: Transform3D
	var parent_bone_idx = hand_skeleton.get_bone_parent(bone_idx)
	if parent_bone_idx != -1:
		parent_global_transform = hand_skeleton.get_bone_global_pose(parent_bone_idx)
	else: # Osso é filho direto do Skeleton3D
		parent_global_transform = hand_skeleton.global_transform

	# Transformar a direção global e o vetor 'up' global para o espaço local do osso pai.
	# A base do pai precisa estar ortonormalizada para uma transposição correta.
	var parent_inv_basis = parent_global_transform.basis.orthonormalized().transposed()
	var direction_local_to_parent = parent_inv_basis * direction_global
	var up_local_to_parent = parent_inv_basis * reference_up_vector_global
	
	# Certificar que up_local_to_parent não é colinear com direction_local_to_parent
	if abs(direction_local_to_parent.dot(up_local_to_parent)) > 0.998:
		# Tentar um 'up' alternativo. Por exemplo, o eixo X local do pai.
		up_local_to_parent = Vector3.RIGHT 
		if abs(direction_local_to_parent.dot(up_local_to_parent)) > 0.998:
			up_local_to_parent = Vector3.FORWARD # Outro fallback

	# Construir a nova base local para o osso.
	var new_local_basis = Basis().looking_at(direction_local_to_parent, up_local_to_parent)
	
	# Usar a translação da pose de descanso (rest pose) do osso.
	# Isso assume que a pose de descanso define corretamente o offset do osso em relação ao seu pai.
	var bone_rest_transform = hand_skeleton.get_bone_rest(bone_idx)
	
	var new_local_transform = Transform3D(new_local_basis.orthonormalized(), bone_rest_transform.origin)

	hand_skeleton.set_bone_pose(bone_idx, new_local_transform)


func set_solid_visible(show_mesh: bool) -> void:
	show_solid = show_mesh
	# if hand_solid_mesh: # Verificar se já foi criado
	# 	 hand_solid_mesh.visible = show_mesh
	# substitui pela lógica do novo modelo
	if rigged_hand_instance:
		rigged_hand_instance.visible = show_mesh

func set_wireframe_visible(show_frame: bool) -> void:
	show_wireframe = show_frame
	for line in hand_lines:
		line.visible = show_frame
	# Os landmarks spheres originais (não os joint_visuals) podem ser controlados aqui se necessário
	for landmark_node in hand_landmarks: # São os HandLandmark, não os MeshInstance
		landmark_node.visible = show_frame # Se quisermos ver os pontos originais

func parse_hand_landmarks_from_data(hand_data: Array) -> void:
	for lm_id in range(NUM_LANDMARKS):
		var lm_data = hand_data[lm_id]
		# Assumindo que hand_data[lm_id] é um Vector3 ou array [x,y,z] em coordenadas de câmara
		var pos_cam: Vector3
		if lm_data is Array and lm_data.size() == 3:
			pos_cam = Vector3(lm_data[0], lm_data[1], lm_data[2])
		elif lm_data is Vector3:
			pos_cam = lm_data
		else:
			printerr("Invalid landmark data format for ID: ", lm_id)
			continue

		pos_cam -= Vector3.ONE * 0.5 # Normalizar
		var pos_xyz = Vector3(-pos_cam.x, -pos_cam.y, pos_cam.z) * HAND_SCALE
		pos_xyz.z += 12 # Ajuste de profundidade arbitrário, pode precisar de calibração
		_update_hand_landmark(lm_id, pos_xyz)

func _update_hand_landmark(landmark_id: int, landmark_pos: Vector3) -> void:
	if landmark_id >= 0 and landmark_id < hand_landmarks.size():
		var lm = hand_landmarks[landmark_id]
		lm.target = landmark_pos # HandLandmark tem uma propriedade target para suavização
	else:
		printerr("Invalid landmark_id: ", landmark_id)

# Remover a função _calculate_bezier_point se não for mais usada em lado nenhum
# func _calculate_bezier_point(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: float) -> Vector3:
#    var t2 = t * t
#    var t3 = t2 * t
#    var mt = 1.0 - t
#    var mt2 = mt * mt
#    var mt3 = mt2 * mt
#    return mt3 * p0 + 3.0 * mt2 * t * p1 + 3.0 * mt * t2 * p2 + t3 * p3
