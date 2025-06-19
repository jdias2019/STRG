extends Node3D

const PORT: int = 4242

# define as ligações entre landmarks da mão para formar esqueleto
const HAND_CONNECTIONS = [
	[0, 1], [1, 2], [2, 3], [3, 4],  # polegar
	[0, 5], [5, 6], [6, 7], [7, 8],  # indicador
	[0, 9], [9, 10], [10, 11], [11, 12],  # médio
	[0, 13], [13, 14], [14, 15], [15, 16],  # anelar
	[0, 17], [17, 18], [18, 19], [19, 20],  # mindinho
	# ligações da palma
	[5, 9], [9, 13], [13, 17], [17, 5],
	# ligações adicionais da palma
	[0, 9], [0, 13], [5, 17]
]

# definições dos segmentos dos dedos
const FINGER_SEGMENTS = {
	"thumb": [[0, 1], [1, 2], [2, 3], [3, 4]],
	"index": [[0, 5], [5, 6], [6, 7], [7, 8]],
	"middle": [[0, 9], [9, 10], [10, 11], [11, 12]],
	"ring": [[0, 13], [13, 14], [14, 15], [15, 16]],
	"pinky": [[0, 17], [17, 18], [18, 19], [19, 20]]
}

var server: UDPServer
var left_hand_geo: Dictionary
var right_hand_geo: Dictionary

# configurações visuais
@export var show_landmarks: bool = true
@export var show_bones: bool = true
@export var show_palm_mesh: bool = true
@export var use_realistic_colors: bool = true
@export var hand_scale: float = 35.0
@export var landmark_size: float = 0.12
@export var bone_thickness: float = 0.06

# suavização e interpolação
@export var target_fps: float = 60.0
@export var interpolation_speed: float = 15.0
@export var position_smoothing: float = 0.7
@export var confidence_threshold: float = 0.5
@export var stability_frames: int = 1

# variáveis de optimização de performance
var last_update_time: float = 0.0
var update_interval: float
var landmark_cache: Dictionary = {}
var transform_cache: Array[Transform3D] = []
var smoothing_cache: Dictionary = {}

# constantes de transformação 3d
var scale_factor: float
var y_inversion: float = -1.0
var center_offset: Vector3 = Vector3(0.5, 0.5, 0)
var depth_enhancement: float = 2.0

# iluminação e ambiente
var main_light: DirectionalLight3D
var environment: Environment

func _ready() -> void:
	server = UDPServer.new()
	server.listen(PORT)
	
	# calcula intervalo de actualização
	update_interval = 1.0 / target_fps
	scale_factor = hand_scale
	
	# pré-aloca caches
	transform_cache.resize(HAND_CONNECTIONS.size())
	for i in range(transform_cache.size()):
		transform_cache[i] = Transform3D()
	
	# inicializa geometrias simples das mãos
	left_hand_geo = _create_simple_hand_geometry("left")
	right_hand_geo = _create_simple_hand_geometry("right")
	
	print("rastreamento simples de mãos inicializado - fps alvo: ", target_fps)

func _setup_3d_environment() -> void:
	"""Configura um ambiente 3D otimizado para visualização de mãos"""
	
	# Add enhanced lighting
	main_light = DirectionalLight3D.new()
	main_light.light_energy = 1.2
	main_light.shadow_enabled = true
	main_light.rotation_degrees = Vector3(-45, 45, 0)
	add_child(main_light)
	
	# Add ambient lighting
	var ambient_light = DirectionalLight3D.new()
	ambient_light.light_energy = 0.3
	ambient_light.rotation_degrees = Vector3(45, -45, 0)
	add_child(ambient_light)
	
	# Setup environment for better 3D rendering
	environment = Environment.new()
	environment.background_mode = Environment.BG_SKY
	environment.sky = Sky.new()
	environment.sky.sky_material = ProceduralSkyMaterial.new()
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	environment.ambient_light_energy = 0.2
	
	# Add subtle fog for depth perception
	environment.fog_enabled = true
	environment.fog_light_color = Color(0.9, 0.95, 1.0)
	environment.fog_light_energy = 0.1
	environment.fog_sun_scatter = 0.1
	
	# Enable SDFGI for better lighting (if supported)
	environment.sdfgi_enabled = true
	environment.sdfgi_use_occlusion = true
	
	# Apply environment to camera (assumes camera exists in scene)
	var camera = get_viewport().get_camera_3d()
	if camera:
		camera.environment = environment

func _create_simple_hand_geometry(hand_type: String) -> Dictionary:
	# cria geometria simples da mão
	
	var hand_color = Color(0.95, 0.75, 0.55) if use_realistic_colors else (Color.CYAN if hand_type == "left" else Color.ORANGE)
	var bone_color = Color(0.85, 0.7, 0.6) if use_realistic_colors else Color.WHITE
	
	var hand_geo = {
		"landmarks": _create_simple_multimesh(21, landmark_size, hand_color, "sphere"),
		"bones": _create_simple_multimesh(HAND_CONNECTIONS.size(), bone_thickness, bone_color, "capsule"),
		"last_positions": [],
		"target_positions": [],
		"stable_frames": 0,
		"hand_type": hand_type
	}
	
	# inicializa arrays de posições
	hand_geo.last_positions.resize(21)
	hand_geo.target_positions.resize(21)
	
	# inicializa todas as posições como null
	for i in range(21):
		hand_geo.last_positions[i] = null
		hand_geo.target_positions[i] = null
	
	_set_hand_visibility(hand_geo, false)
	return hand_geo

func _create_simple_multimesh(count: int, size: float, color: Color, mesh_type: String) -> MultiMeshInstance3D:
	# cria multimesh simples
	
	var multimesh_instance = MultiMeshInstance3D.new()
	multimesh_instance.multimesh = MultiMesh.new()
	multimesh_instance.multimesh.transform_format = MultiMesh.TRANSFORM_3D
	multimesh_instance.multimesh.instance_count = count
	
	# cria mesh 3d baseado no tipo
	var mesh: Mesh
	match mesh_type:
		"sphere":
			var sphere_mesh = SphereMesh.new()
			sphere_mesh.radius = size
			sphere_mesh.height = size * 2
			sphere_mesh.radial_segments = 8
			sphere_mesh.rings = 6
			mesh = sphere_mesh
		"capsule":
			var capsule_mesh = CapsuleMesh.new()
			capsule_mesh.radius = size
			capsule_mesh.height = 1.0
			capsule_mesh.radial_segments = 6
			capsule_mesh.rings = 3
			mesh = capsule_mesh
		_:
			mesh = SphereMesh.new()
	
	# cria material 3d simples
	var material = StandardMaterial3D.new()
	material.albedo_color = color
	material.metallic = 0.0
	material.roughness = 0.6
	material.specular = 0.3
	material.rim_enabled = true
	material.rim = 0.3
	material.rim_tint = 0.2
	
	# material simplificado para melhor performance
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.diffuse_mode = BaseMaterial3D.DIFFUSE_LAMBERT
	
	multimesh_instance.multimesh.mesh = mesh
	multimesh_instance.material_override = material
	multimesh_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	
	add_child(multimesh_instance)
	return multimesh_instance

func _create_palm_mesh(color: Color) -> MeshInstance3D:
	"""Cria uma malha 3D para a palma da mão"""
	
	var palm_mesh_instance = MeshInstance3D.new()
	
	# Create a simple quad mesh for the palm
	var array_mesh = ArrayMesh.new()
	var arrays = []
	arrays.resize(Mesh.ARRAY_MAX)
	
	# Define palm vertices (will be updated dynamically)
	var vertices = PackedVector3Array()
	var normals = PackedVector3Array()
	var uvs = PackedVector2Array()
	var indices = PackedInt32Array()
	
	# Create a simple quad for now (will be updated with actual palm shape)
	vertices.push_back(Vector3(-1, 0, -1))
	vertices.push_back(Vector3(1, 0, -1))
	vertices.push_back(Vector3(1, 0, 1))
	vertices.push_back(Vector3(-1, 0, 1))
	
	for i in range(4):
		normals.push_back(Vector3.UP)
		uvs.push_back(Vector2(i % 2, i / 2))
	
	indices.append_array([0, 1, 2, 0, 2, 3])
	
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = indices
	
	array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	palm_mesh_instance.mesh = array_mesh
	
	# Enhanced palm material
	var material = StandardMaterial3D.new()
	material.albedo_color = color
	material.albedo_color.a = 0.7  # Semi-transparent
	material.metallic = 0.0
	material.roughness = 0.4
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.cull_mode = BaseMaterial3D.CULL_DISABLED  # Show both sides
	
	palm_mesh_instance.material_override = material
	palm_mesh_instance.visible = show_palm_mesh
	
	add_child(palm_mesh_instance)
	return palm_mesh_instance

func _initialize_smoothing_data() -> void:
	"""Inicializa dados de suavização para ambas as mãos"""
	smoothing_cache["left"] = {
		"velocity": [],
		"acceleration": [],
		"last_time": 0.0
	}
	smoothing_cache["right"] = {
		"velocity": [],
		"acceleration": [],
		"last_time": 0.0
	}
	
	for hand_type in ["left", "right"]:
		smoothing_cache[hand_type].velocity.resize(21)
		smoothing_cache[hand_type].acceleration.resize(21)
		for i in range(21):
			smoothing_cache[hand_type].velocity[i] = Vector3.ZERO
			smoothing_cache[hand_type].acceleration[i] = Vector3.ZERO

func _set_hand_visibility(hand_geo: Dictionary, p_visible: bool) -> void:
	# controla visibilidade dos elementos da mão
	hand_geo.landmarks.visible = show_landmarks and p_visible
	hand_geo.bones.visible = show_bones and p_visible

func _parse_hands_from_packet(data: PackedByteArray) -> Dictionary:
	# parse optimizado com cache
	var json_string = data.get_string_from_utf8()
	
	# usa cache com limite de tamanho
	if landmark_cache.has(json_string):
		return landmark_cache[json_string]
	
	var json = JSON.new()
	var error = json.parse(json_string)
	if error == OK and typeof(json.data) == TYPE_DICTIONARY:
		landmark_cache[json_string] = json.data
		# gestão agressiva de cache
		if landmark_cache.size() > 20:
			# remove entradas antigas
			var keys = landmark_cache.keys()
			for i in range(10):
				landmark_cache.erase(keys[i])
		return json.data
	return {}

func _process(delta: float) -> void:
	# alterna controlos de visibilidade
	if Input.is_action_just_pressed("ui_focus_next"):
		show_landmarks = !show_landmarks
		_set_hand_visibility(left_hand_geo, true)
		_set_hand_visibility(right_hand_geo, true)

	# interpolação simples
	_interpolate_hands(delta)
	
	# processa pacotes udp
	server.poll()
	if server.is_connection_available():
		var peer = server.take_connection()
		var data = peer.get_packet()
		var hands_data = _parse_hands_from_packet(data)

		_update_hand_data(left_hand_geo, hands_data.get("left"))
		_update_hand_data(right_hand_geo, hands_data.get("right"))

func _interpolate_hands(delta: float) -> void:
	# interpolação simples
	_interpolate_hand(left_hand_geo, delta)
	_interpolate_hand(right_hand_geo, delta)

func _interpolate_hand(hand_geo: Dictionary, delta: float) -> void:
	# interpolação simples para uma mão
	if hand_geo.last_positions.size() == 0 or hand_geo.target_positions.size() == 0:
		return
		
	var landmarks_multimesh = hand_geo.landmarks.multimesh
	var bones_multimesh = hand_geo.bones.multimesh
	
	# interpolação simples
	for i in range(21):
		if hand_geo.last_positions[i] != null and hand_geo.target_positions[i] != null:
			var current_pos = hand_geo.last_positions[i]
			var target_pos = hand_geo.target_positions[i]
			var new_pos = current_pos.lerp(target_pos, interpolation_speed * delta)
			
			hand_geo.last_positions[i] = new_pos
			landmarks_multimesh.set_instance_transform(i, Transform3D(Basis(), new_pos))
	
	# actualiza ligações dos ossos
	_update_bone_connections(hand_geo, bones_multimesh)

func _update_hand_data(hand_geo: Dictionary, hand_data) -> void:
	# actualização simples dos dados da mão
	if hand_data:
		_set_hand_visibility(hand_geo, true)
		_update_hand_geometry(hand_geo, hand_data)
	else:
		_set_hand_visibility(hand_geo, false)

func _update_hand_geometry(hand_geo: Dictionary, hand_data: Dictionary) -> void:
	# actualização simples da geometria da mão
	var landmarks_multimesh = hand_geo.landmarks.multimesh

	# actualiza posições alvo
	for i in range(21):
		if hand_data.has(str(i)):
			var pos_array = hand_data[str(i)]
			var raw_pos = Vector3(pos_array[0], pos_array[1], pos_array[2])
			
			var transformed_pos = Vector3(
				(raw_pos.x - center_offset.x) * scale_factor,
				(raw_pos.y - center_offset.y) * scale_factor * y_inversion,
				raw_pos.z * scale_factor
			)
			
			hand_geo.target_positions[i] = transformed_pos
			
			# inicializa posições se não estiverem definidas
			if hand_geo.last_positions[i] == null:
				hand_geo.last_positions[i] = transformed_pos
				landmarks_multimesh.set_instance_transform(i, Transform3D(Basis(), transformed_pos))

func _update_bone_connections(hand_geo: Dictionary, bones_multimesh: MultiMesh) -> void:
	# actualização simples das ligações ósseas
	var positions = hand_geo.last_positions
	
	for i in range(HAND_CONNECTIONS.size()):
		var connection = HAND_CONNECTIONS[i]
		var start_pos = positions[connection[0]]
		var end_pos = positions[connection[1]]

		if start_pos != null and end_pos != null:
			var diff = end_pos - start_pos
			var distance = diff.length()
			
			if distance > 0.001:
				var center = start_pos + diff * 0.5
				var y_axis = diff / distance
				var z_axis: Vector3
				
				if abs(y_axis.dot(Vector3.UP)) < 0.99:
					z_axis = y_axis.cross(Vector3.UP).normalized()
				else:
					z_axis = y_axis.cross(Vector3.RIGHT).normalized()
				var x_axis = y_axis.cross(z_axis)
				
				var scale_vec = Vector3(bone_thickness * 2, distance, bone_thickness * 2)
				var basis = Basis(x_axis * scale_vec.x, y_axis * scale_vec.y, z_axis * scale_vec.z)
				
				transform_cache[i] = Transform3D(basis, center)
				bones_multimesh.set_instance_transform(i, transform_cache[i])

func _update_palm_mesh(hand_geo: Dictionary) -> void:
	"""Atualiza a malha da palma com base nos landmarks da mão"""
	var palm_mesh = hand_geo.palm_mesh
	var positions = hand_geo.smoothed_positions
	
	# Define palm landmarks (wrist and base of fingers)
	var palm_landmarks = [0, 5, 9, 13, 17]  # Wrist and finger bases
	var valid_positions = []
	
	for landmark_idx in palm_landmarks:
		if positions[landmark_idx] != null:
			valid_positions.append(positions[landmark_idx])
	
	# Only update if we have enough valid positions
	if valid_positions.size() >= 4:
		# Calculate palm center and normal
		var palm_center = Vector3.ZERO
		for pos in valid_positions:
			palm_center += pos
		palm_center /= valid_positions.size()
		
		# Update palm mesh position and orientation
		palm_mesh.position = palm_center
		
		# Calculate palm orientation based on landmarks
		if valid_positions.size() >= 3:
			var v1 = valid_positions[1] - valid_positions[0]
			var v2 = valid_positions[2] - valid_positions[0]
			var normal = v1.cross(v2).normalized()
			
			# Create transform that aligns palm with calculated normal
			var transform = Transform3D()
			transform.origin = palm_center
			transform.basis = transform.basis.looking_at(normal, Vector3.UP)
			palm_mesh.transform = transform
