extends Node3D

const PORT: int = 4242

var server: UDPServer

var left_hand: Hand
var right_hand: Hand

# configurações de visualização
@export var show_wireframe: bool = false  # por padrão, não mostra o wireframe
@export var show_solid: bool = true  # por padrão, mostra a malha sólida

func _ready() -> void:
	server = UDPServer.new()
	server.listen(PORT)
	
	left_hand = _create_new_hand()
	right_hand = _create_new_hand()
	
	# configurando a visualização das mãos
	_update_hand_visualization()

func _create_new_hand() -> Hand:
	var hand_instance := Hand.new()
	add_child(hand_instance)
	return hand_instance

# atualiza a visualização das mãos com base nas configurações
func _update_hand_visualization() -> void:
	if left_hand:
		left_hand.set_wireframe_visible(show_wireframe)
		left_hand.set_solid_visible(show_solid)
	
	if right_hand:
		right_hand.set_wireframe_visible(show_wireframe)
		right_hand.set_solid_visible(show_solid)

func _parse_hands_from_packet(data: PackedByteArray) -> Dictionary:
	var json_string = data.get_string_from_utf8()
	var json = JSON.new()
	
	var error = json.parse(json_string)
	assert(error == OK)
	
	var data_received = json.data
	assert(typeof(data_received) == TYPE_DICTIONARY)
	
	return data_received

func _process(_delta: float) -> void:
	server.poll()
	if server.is_connection_available():
		var peer = server.take_connection()
		var data = peer.get_packet()
		var hands_data = _parse_hands_from_packet(data)
		
		if hands_data["left"] != null:
			left_hand.parse_hand_landmarks_from_data(hands_data["left"])
			
		if hands_data["right"] != null:
			right_hand.parse_hand_landmarks_from_data(hands_data["right"])
	
	# permite alternar a visualização do wireframe com a tecla W
	if Input.is_action_just_pressed("ui_focus_next"):  # Tecla Tab
		show_wireframe = !show_wireframe
		_update_hand_visualization()
