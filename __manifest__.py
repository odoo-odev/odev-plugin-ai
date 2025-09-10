"""Plugin description."""

# --- Version information ------------------------------------------------------
#
# Version number breakdown: <major>.<minor>.<patch>
#
# major:  Major version number, incremented when a new major feature is added
#         when important changes are made to the framework or when backwards
#         compatibility is broken.
# minor:  Minor version number, incremented when a new minor feature is added
#         that does not break backwards compatibility. This may indicate
#         additions of new commands or new features to existing commands.
#         This number is reset to 0 when the major version number is
#         incremented.
# patch:  Patch version number, incremented when a bug is fixed or when
#         documentation is updated. May also be incremented when a new
#         migration script is added.
#         This number is reset to 0 when the minor version number is
#         incremented.
#
# Version number should be incremented once and only once per pull request
# or merged change.
# ------------------------------------------------------------------------------

__version__ = "1.0.0"

# --- Dependencies -------------------------------------------------------------
# List other odev plugins from which this current plugin depends.
# Those dependent plugins will be loaded first, allowing this one to inject
# code into existing commands to empower them.
#
# The format for listing a plugin is the same as the one expected in the plugin
# command: "<github organization>/<repository name>".
#
# All plugins depend from odev core by default.
# ------------------------------------------------------------------------------

depends = []
