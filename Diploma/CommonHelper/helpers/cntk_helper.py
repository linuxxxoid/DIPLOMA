import cntk as C

#region Загрузка слоев из модели (и несколько функций для понимаяни что за слои в модели)
def get_layers_by_layers(graph, first_layer, last_layer, is_freeze = True):
    return C.combine([last_layer.owner]).clone( \
        C.CloneMethod.freeze if is_freeze  else C.CloneMethod.clone, \
        { first_layer : C.placeholder()  })   

def get_layers_by_indexes(graph, index_first_layer, index_last_layer, is_freeze = True):
    # C.logging.get_node_outputs(graph)
    layers_without_name =  C.logging.graph.find_all_with_name(graph, '')
    return get_layers_by_layers(graph, layers_without_name[index_first_layer], layers_without_name[index_last_layer], is_freeze )

def get_layers_by_name_index(graph, name_first_layer, index_last_layer, is_freeze = True):
    layers_without_name =  C.logging.graph.find_all_with_name(graph, '')
    return get_layers_by_layers(graph, C.logging.find_by_name(graph, name_first_layer), layers_without_name[index_last_layer], is_freeze)
    
def get_layers_by_index_name(graph, index_first_layer, name_last_layer, is_freeze = True):
    layers_without_name =  C.logging.graph.find_all_with_name(graph, '')
    return get_layers_by_layers(graph, layers_without_name[index_first_layer], C.logging.find_by_name(graph, name_last_layer), is_freeze)

def get_layers_by_names(graph, name_first_layer, name_last_layer, is_freeze = True):
    return get_layers_by_layers(graph, C.logging.find_by_name(graph, name_first_layer), C.logging.find_by_name(graph, name_last_layer), is_freeze)

def get_layers_by_uid_name(graph, uid_first_layer, name_last_layer, is_freeze = True):
    return get_layers_by_layers(graph, C.logging.find_by_uid(graph, uid_first_layer), C.logging.find_by_name(graph, name_last_layer), is_freeze)

def get_layers_by_name_uid(graph, name_first_layer, uid_last_layer, is_freeze = True):
    return get_layers_by_layers(graph, C.logging.find_by_name(graph, name_first_layer), C.logging.find_by_uid(graph, uid_last_layer), is_freeze)

def get_layers_by_uids(graph, uid_first_layer, uid_last_layer, is_freeze = True):
    return get_layers_by_layers(graph, C.logging.find_by_uid(graph, uid_first_layer), C.logging.find_by_uid(graph, uid_last_layer), is_freeze)

def get_layers(graph):
    return C.logging.get_node_outputs(graph)

def print_layers(graph):
    list_layers = get_layers(graph)
    [print('{0} : {1}, {2}, {3}'.format(i, x.name, x.uid, x.shape)) for i, x in enumerate(list_layers)]
    print('\r\n')

# TODO
def print_layers_without_name():
    # layers_without_name =  C.logging.graph.find_all_with_name(graph, '')
    pass


def create_graph_dot(graph, name):
    pass

#endregion 
