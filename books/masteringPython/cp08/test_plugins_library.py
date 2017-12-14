# test the plugins library in plugins directory
import plugins

# test load method
plugins.Plugins.load(
    'plugins.spam',
    'plugins.eggs',
)
# test get method
plugins.Plugins.get('spam')
plugins.Plugins.get('spam')
