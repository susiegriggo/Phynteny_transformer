
import argparse

class AdvancedArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with support for basic/advanced options."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._basic_options = []
        self._advanced_options = []
        self._advanced_mode = False
    
    def add_basic_argument(self, *args, **kwargs):
        """Add an argument that's always visible."""
        action = self.add_argument(*args, **kwargs)
        self._basic_options.append(action.dest)
        return action
    
    def add_advanced_argument(self, *args, **kwargs):
        """Add an argument that's only visible in advanced mode."""
        action = self.add_argument(*args, **kwargs)
        self._advanced_options.append(action.dest)
        return action
    
    def parse_args(self, args=None, namespace=None):
        # First, see if --advanced is in the args
        temp_parser = argparse.ArgumentParser(add_help=False)
        temp_parser.add_argument('--advanced', action='store_true')
        temp_args, _ = temp_parser.parse_known_args(args)
        self._advanced_mode = temp_args.advanced

        # Add the --advanced flag to the main parser
        self.add_argument('--advanced', action='store_true', 
                         help='Show advanced options for custom models')
        
        # Parse arguments normally
        return super().parse_args(args, namespace)
    
    def format_help(self):
        """Custom help formatter that only shows relevant options."""
        if not self._advanced_mode:
            # Store the original option strings and help text for advanced options
            backup = {}
            for action in self._actions:
                if action.dest in self._advanced_options and action.dest != 'advanced':
                    backup[action.dest] = (action.help, action.option_strings)
                    action.help = argparse.SUPPRESS
                    action.option_strings = []
            
            # Get the help text
            help_text = super().format_help()
            
            # Restore the original options
            for action in self._actions:
                if action.dest in backup:
                    action.help, action.option_strings = backup[action.dest]
            
            return help_text + "\n\nFor advanced options, use --advanced flag."
        else:
            return super().format_help()