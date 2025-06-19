_api_key = None
_username = None
_user_already_set = False
_project_name = None

def set_comet_user(username):
    global _user_already_set, _api_key, _username
    _api_keys = {"Marina" : "Insert your API key here!"}
    if _user_already_set:
        raise Exception("User already set")
    if not username in _api_keys.keys():
        raise Exception(f"Invalid username, we expect one of these names: {set(_api_keys.keys())}.")
    print(f"You are setting the comet user to '{username}'. Type 'Enter' to confirm or 'q' to quit:")

    valid_input = False
    while not valid_input:
        answer = input()
        if answer == 'q':
            raise Exception('User quit')
        elif answer == "":
            valid_input = True
        else:
            print("Invalid answer. Type 'Enter' to confirm or 'q' to quit:")

    _user_already_set = True
    _username = username
    _api_key = _api_keys.get(username)

def set_project_name(project_name):
    global _project_name
    if not _user_already_set:
        raise Exception("You must first set the user!")

    import comet_ml

    comet_ml.login(api_key=_api_key)
    api = comet_ml.API()
    workspaces = api.get()
    for workspace in workspaces:
        projects = api.get(workspace)  # Returns a list of project names in this workspace[1][5]
        if project_name in projects:
            print(f"Project '{project_name}' already exists in workspace '{workspace}' of user '{_username}'.")
            print("Do you want to continue? Type 'Enter' to confirm or 'q' to quit:")

            valid_input = False
            while not valid_input:
                answer = input()
                if answer == 'q':
                    raise Exception('User quit')
                elif answer == "":
                    valid_input = True
                else:
                    print("Invalid answer. Type 'Enter' to confirm or 'q' to quit:")

    _project_name = project_name

def set_comet_user_and_project_name(username, project_name):
    set_comet_user(username)
    set_project_name(project_name)

def get_api_key():
    return _api_key

def get_project_name():
    return _project_name

if __name__ == "__main__":
    # Example for valid user and existing project
    set_comet_user_and_project_name("Martin", "6-s191lab2-part1-cnn")