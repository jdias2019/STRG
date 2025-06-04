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
	_create_hand_landmark_spheres() # Isto cria os HandLandmark (Node3D com script)
	_create_hand_lines_visuals() # Para o modo wireframe
	_create_solid_hand_visuals() # Nova função para criar as malhas sólidas uma vez

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
	hand_solid_mesh = Node3D.new() # Usar Node3D como um contentor simples
	hand_solid_mesh.name = "SolidHandVisuals"
	add_child(hand_solid_mesh)

	mesh_material = StandardMaterial3D.new()
	mesh_material.albedo_color = Color(0.95, 0.88, 0.82, 1.0)
	mesh_material.roughness = 0.75
	mesh_material.metallic_specular = 0.15
	mesh_material.cull_mode = StandardMaterial3D.CULL_BACK
	mesh_material.shading_mode = StandardMaterial3D.SHADING_MODE_PER_VERTEX
	mesh_material.subsurf_scatter_enabled = true
	mesh_material.subsurf_scatter_strength = 0.5
	mesh_material.subsurf_scatter_skin_mode = true
	# hand_solid_mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON # Não aplicável a Node3D, aplicar às MeshInstances

	# Criar malhas para as juntas (esferas)
	for i in range(NUM_LANDMARKS):
		var sphere_inst = MeshInstance3D.new()
		var sphere_m = SphereMesh.new()
		# Raio e segmentos serão definidos em _update_hand_solid_mesh com base no tipo de junta
		sphere_m.radial_segments = segments_medium # Valor base
		sphere_m.rings = segments_medium / 2       # Valor base
		sphere_inst.mesh = sphere_m
		sphere_inst.material_override = mesh_material
		sphere_inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
		hand_solid_mesh.add_child(sphere_inst)
		joint_visuals.append(sphere_inst)

	# Criar malhas para os ossos (cápsulas)
	for i in range(HAND_LINES_MAPPING.size()):
		var capsule_inst = MeshInstance3D.new()
		var capsule_m = CapsuleMesh.new()
		capsule_m.radius = bone_capsule_radius # Raio é mais constante para ossos
		capsule_m.radial_segments = segments_medium
		capsule_m.rings = segments_medium / 2 # Para as extremidades da cápsula
		capsule_inst.mesh = capsule_m
		capsule_inst.material_override = mesh_material
		capsule_inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
		hand_solid_mesh.add_child(capsule_inst)
		bone_visuals.append(capsule_inst)
	
	hand_solid_mesh.visible = show_solid


func _process(_delta: float) -> void:
	if show_wireframe:
		_update_hand_lines_visuals()
	
	if show_solid and hand_solid_mesh.visible: # Apenas atualiza se estiver visível
		_update_hand_solid_mesh()

func _update_hand_lines_visuals() -> void: # Renomeado para clareza
	for i in range(HAND_LINES_MAPPING.size()):
		var mapping = HAND_LINES_MAPPING[i]
		var p0 = hand_landmarks[mapping[0]].global_position
		var p1 = hand_landmarks[mapping[1]].global_position
		LineRenderer.edit_line(hand_lines[i], p0, p1)


func _update_hand_solid_mesh() -> void:
	# Atualizar juntas (esferas)
	for i in range(NUM_LANDMARKS):
		var sphere_inst = joint_visuals[i]
		var sphere_m = sphere_inst.mesh as SphereMesh # Cast para aceder a propriedades do SphereMesh
		
		var current_joint_radius = joint_radius_base
		var current_segments = segments_high

		if i in [0, 5, 9, 13, 17]: # Pulso e base dos dedos
			current_joint_radius = joint_radius_base
			current_segments = segments_high
		elif i in [4, 8, 12, 16, 20]: # Pontas dos dedos
			current_joint_radius = joint_radius_tip
			current_segments = segments_medium
		else: # Juntas intermédias dos dedos
			current_joint_radius = joint_radius_finger
			current_segments = segments_medium
		
		sphere_m.radius = current_joint_radius
		sphere_m.height = current_joint_radius * 2.0 # Diâmetro
		sphere_m.radial_segments = current_segments
		sphere_m.rings = int(current_segments / 2) # Assegurar int
		
		sphere_inst.global_position = hand_landmarks[i].global_position

	# Atualizar ossos (cápsulas)
	# var current_bone_capsule_radius = bone_capsule_radius # Já é uma var de membro
	var current_bone_segments = segments_medium

	for i in range(HAND_LINES_MAPPING.size()):
		var capsule_inst = bone_visuals[i]
		var capsule_m = capsule_inst.mesh as CapsuleMesh # Cast
		
		capsule_m.radius = bone_capsule_radius
		capsule_m.radial_segments = current_bone_segments
		capsule_m.rings = int(current_bone_segments / 2) # Assegurar int

		var bone_map_indices = HAND_LINES_MAPPING[i]
		var p0_idx = bone_map_indices[0]
		var p1_idx = bone_map_indices[1]

		var p0 = hand_landmarks[p0_idx].global_position
		var p1 = hand_landmarks[p1_idx].global_position

		var distance = p0.distance_to(p1)
		
		if distance < 0.001: # Evitar cápsulas de comprimento zero ou muito pequeno
			capsule_inst.visible = false
			continue
		else:
			capsule_inst.visible = true
		
		# Assegurar que a altura da cápsula não é menor que o diâmetro (resulta em artefactos)
		# A altura da CapsuleMesh é a altura total, incluindo as meias-esferas das pontas.
		# A parte cilíndrica tem altura = mesh.height - 2 * mesh.radius
		var capsule_total_height = distance # A distância entre centros das esferas das juntas
		# Se a cápsula liga duas esferas, a sua altura deve ser a distância entre as superfícies,
		# mas CapsuleMesh define altura total. Para cobrir a distância entre os centros p0 e p1,
		# a altura da cápsula deve ser essa distância.
		
		capsule_m.height = capsule_total_height
		
		capsule_inst.global_position = (p0 + p1) / 2.0
		
		var y_axis = (p1 - p0).normalized()
		var x_axis: Vector3
		
		var dot_y_up = y_axis.dot(Vector3.UP)
		if abs(dot_y_up) > 0.999: # Quase paralelo ao eixo Y global
			x_axis = y_axis.cross(Vector3.RIGHT).normalized() # Usar eixo X para cross product
		else:
			x_axis = y_axis.cross(Vector3.UP).normalized()
		
		# Verificar se x_axis é válido (não zero)
		if x_axis.length_squared() < 0.0001:
			if abs(y_axis.dot(Vector3.RIGHT)) > 0.999: # Quase paralelo ao eixo X global
				x_axis = y_axis.cross(Vector3.FORWARD).normalized()
			else:
				x_axis = y_axis.cross(Vector3.RIGHT).normalized()

		var z_axis = x_axis.cross(y_axis).normalized()
		
		capsule_inst.transform.basis = Basis(x_axis, y_axis, z_axis)


func set_solid_visible(show_mesh: bool) -> void:
	show_solid = show_mesh
	if hand_solid_mesh: # Verificar se já foi criado
		hand_solid_mesh.visible = show_mesh

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
