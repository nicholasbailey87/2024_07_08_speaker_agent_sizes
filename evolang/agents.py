from __future__ import annotations
import copy

class Opt(dict):
    """
    Class for tracking options.

    Functions like a dict, but allows us to track the history of arguments as they are
    set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.deepcopies = []

    def __setitem__(self, key, val):
        loc = traceback.format_stack(limit=2)[-2]
        self.history.append((key, val, loc))
        super().__setitem__(key, val)

    def __getstate__(self):
        return (self.history, self.deepcopies, dict(self))

    def __setstate__(self, state):
        self.history, self.deepcopies, data = state
        self.update(data)

    def __reduce__(self):
        return (Opt, (), self.__getstate__())

    def __deepcopy__(self, memo):
        """
        Override deepcopy so that history is copied over to new object.
        """
        # track location of deepcopy
        loc = traceback.format_stack(limit=3)[-3]
        self.deepcopies.append(loc)
        # copy all our children
        memo = Opt({k: copy.deepcopy(v) for k, v in self.items()})
        # deepcopy the history. history is only tuples, so we can do it shallow
        memo.history = copy.copy(self.history)
        # deepcopy the list of deepcopies. also shallow bc only strings
        memo.deepcopies = copy.copy(self.deepcopies)
        return memo

    def display_deepcopies(self):
        """
        Display all deepcopies.
        """
        if len(self.deepcopies) == 0:
            return 'No deepcopies performed on this opt.'
        return '\n'.join(f'{i}. {loc}' for i, loc in enumerate(self.deepcopies, 1))

    def display_history(self, key):
        """
        Display the history for an item in the dict.
        """
        changes = []
        i = 0
        for key_, val, loc in self.history:
            if key != key_:
                continue
            i += 1
            changes.append(f'{i}. {key} was set to {val} at:\n{loc}')
        if changes:
            return '\n'.join(changes)
        else:
            return f'No history for {key}'

    def save(self, filename: str) -> None:
        """
        Save the opt to disk.

        Attempts to 'clean up' any residual values automatically.
        """
        # start with a shallow copy
        dct = dict(self)

        # clean up some things we probably don't want to save
        for key in __AUTOCLEAN_KEYS__:
            if key in dct:
                del dct[key]

        with PathManager.open(filename, 'w', encoding='utf-8') as f:
            json.dump(dct, fp=f, indent=4)
            # extra newline for convenience of working with jq
            f.write('\n')

    @classmethod
    def load(cls, optfile: str) -> Opt:
        """
        Load an Opt from disk.
        """
        try:
            # try json first
            with PathManager.open(optfile, 'r', encoding='utf-8') as t_handle:
                dct = json.load(t_handle)
        except UnicodeDecodeError:
            # oops it's pickled
            with PathManager.open(optfile, 'rb') as b_handle:
                dct = pickle.load(b_handle)
        for key in __AUTOCLEAN_KEYS__:
            if key in dct:
                del dct[key]
        return cls(dct)

    @classmethod
    def load_init(cls, optfile: str) -> Opt:
        """
        Like load, but also looks in opt_presets folders.

        optfile may also be a comma-separated list of multiple presets/files.
        """
        if "," in optfile:
            # load and combine each of the individual files
            new_opt = cls()
            for subopt in optfile.split(","):
                new_opt.update(cls.load_init(subopt))
            return new_opt

        oa_filename = os.path.join("opt_presets", optfile + ".opt")
        user_filename = os.path.join(os.path.expanduser(f"~/.parlai"), oa_filename)
        if PathManager.exists(optfile):
            return cls.load(optfile)
        elif PathManager.exists(user_filename):
            # use a user's custom opt preset
            return cls.load(user_filename)
        else:
            # Maybe a bundled opt preset
            for root in ['parlai', 'parlai_internal', 'parlai_fb']:
                try:
                    if pkg_resources.resource_exists(root, oa_filename):
                        return cls.load(
                            pkg_resources.resource_filename(root, oa_filename)
                        )
                except ModuleNotFoundError:
                    continue

        # made it through without a return path so raise the error
        raise FileNotFoundError(
            f"Could not find filename '{optfile} or opt preset '{optfile}.opt'. "
            "Please check https://parl.ai/docs/opt_presets.html for a list "
            "of available opt presets."
        )

    def log(self, header="Opt"):
        from parlai.core.params import print_git_commit

        logging.info(header + ":")
        for key in sorted(self.keys()):
            valstr = str(self[key])
            if valstr.replace(" ", "").replace("\n", "") != valstr:
                # show newlines as escaped keys, whitespace with quotes, etc
                valstr = repr(valstr)
            logging.info(f"    {key}: {valstr}")
        print_git_commit()

class Agent(object):
    """
    Base class for all other agents.
    """

    def __init__(self, opt: Opt, shared=None):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None

    def observe(self, observation):
        """
        Receive an observation/action dict.
        """
        self.observation = observation
        return observation

    def act(self):
        """
        Return an observation/action dict based upon given observation.
        """
        if hasattr(self, 'observation') and self.observation is not None:
            logging.info(f'agent received observation:\n{self.observation}')

        t = {}
        t['text'] = 'hello, teacher!'
        logging.info(f'agent sending message:\n{t}')
        return t

    def getID(self):
        """
        Return the agent ID.
        """
        return self.id

    def epoch_done(self):
        """
        Return whether the epoch is done or not.

        :rtype: boolean
        """
        return False

    def reset(self):
        """
        Reset the agent, clearing its observation.

        Many subclasses implement additional reset logic.
        """
        self.observation = None

    def reset_metrics(self):
        """
        Reset any metrics reported by this agent.

        This is called to indicate metrics should start fresh, and is typically called
        between loggings or after a `report()`.
        """
        pass

    def save(self, path=None):
        """
        Save any parameters needed to recreate this agent from loaded parameters.

        Default implementation is no-op, but many subagents implement this logic.
        """
        pass

    def clone(self):
        """
        Make a shared copy of this agent.

        Should be the same as using create_agent_from_shared(.), but slightly easier.
        """
        return type(self)(self.opt, self.share())

    def share(self):
        """
        Share any parameters needed to create a shared version of this agent.

        Default implementation shares the class and the opt, but most agents will want
        to also add model weights, teacher data, etc. This especially useful for
        avoiding providing pointers to large objects to all agents in a batch.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        return shared

    def shutdown(self):
        """
        Perform any final cleanup if needed.
        """
        pass

    def respond(
        self, text_or_message: Union[str, Message], **other_message_fields
    ) -> str:
        """
        An agent convenience function which calls the act() and provides a string
        response to a text or message field.

        :param Union[str, Message] text_or_message:
            A string for the 'text' field or a message which MUST
            comprise of the 'text' field apart from other fields.
        :param kwargs other_message_fields:
            Provide fields for the message in the form of keyword arguments.
        :return:
            Agent's response to the message.
        :rtype:
            str
        """
        if isinstance(text_or_message, str):
            observation = Message(text=text_or_message, **other_message_fields)
        else:
            observation = Message(**text_or_message, **other_message_fields)
            if 'text' not in observation:
                raise RuntimeError('The agent needs a \'text\' field in the message.')

        if 'episode_done' not in observation:
            observation['episode_done'] = True
        agent = self.clone()
        agent.observe(observation)
        response = agent.act()
        return response['text']

    def batch_respond(self, messages: List[Message]) -> List[str]:
        """
        An agent convenience function which calls the batch_act() and provides a batch
        response to a list of messages.

        :param List[Message] messages:
            A list of messages each of which MUST comprise of the 'text' field
            apart from other fields.
        :return:
            Agent's batch response to the messages.
        :rtype:
            List[str]
        """
        observations = []
        agents = []
        for i, message in enumerate(messages):
            if 'text' not in message:
                raise RuntimeError(
                    'The agent needs a \'text\' field in the {}th message.'.format(i)
                )
            if 'episode_done' not in message:
                message['episode_done'] = True
            agent = self.clone()
            agents.append(agent)
            observations.append(agent.observe(message))
        agent_acts = self.batch_act(observations)
        response = []
        for agent, resp in zip(agents, agent_acts):
            if hasattr(agent, "self_observe"):
                agent.self_observe(resp)
            response.append(resp['text'])
        return response

    @classmethod
    def upgrade_opt(cls, opt_from_disk: Opt):
        """
        Upgrade legacy options when loading an opt file from disk.

        This is primarily made available to provide a safe space to handle
        backwards-compatible behavior. For example, perhaps we introduce a
        new option today, which wasn't previously available. We can have the
        argument have a new default, but fall back to the "legacy" compatibility
        behavior if the option doesn't exist.

        ``upgrade_opt`` provides an opportunity for such checks for backwards
        compatibility. It is called shortly after loading the opt file from
        disk, and is called before the Agent is initialized.

        Other possible examples include:

            1. Renaming an option,
            2. Deprecating an old option,
            3. Splitting coupled behavior, etc.

        Implementations of ``upgrade_opt`` should conform to high standards,
        due to the risk of these methods becoming complicated and difficult to
        reason about. We recommend the following behaviors:

            1. ``upgrade_opt`` should only be used to provide backwards
            compatibility.  Other behavior should find a different location.
            2. Children should always call the parent's ``upgrade_opt`` first.
            3. ``upgrade_opt`` should always warn when an option was overwritten.
            4. Include comments annotating the date and purpose of each upgrade.
            5. Add an integration test which ensures your old work behaves
            appropriately.

        :param Opt opt_from_disk:
            The opt file, as loaded from the ``.opt`` file on disk.
        :return:
            The modified options
        :rtype:
            Opt
        """
        # 2019-07-11: currently a no-op.
        return opt_from_disk