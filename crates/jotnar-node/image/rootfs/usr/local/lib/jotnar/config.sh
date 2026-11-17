# JotnarOS node configuration loader.
#
# Sourced by /init and /usr/local/sbin/jotnar-service. The build-time defaults
# live in /etc/jotnar/node.conf; a persistent override can live on the data
# partition at /var/lib/jotnar/etc/node.conf and is applied last.

jotnar_load_config() {
    JOTNAR_HOSTNAME=jotnar-node
    JOTNAR_BIND=0.0.0.0
    JOTNAR_PORT=9999
    JOTNAR_ALLOW=lattice
    JOTNAR_MAX_FRAME=67108864
    JOTNAR_TOKEN=
    JOTNAR_TOKEN_FILE=/etc/jotnar/node.token
    JOTNAR_DATA_DIR=/var/lib/jotnar
    JOTNAR_LOG_DIR=/var/lib/jotnar/log
    JOTNAR_CONSOLE=both
    JOTNAR_SSH=0
    JOTNAR_SSH_PORT=22
    JOTNAR_SUPERVISOR=0
    JOTNAR_OSD=0
    JOTNAR_MODULE_SOCKET=
    JOTNAR_CONSOLE_AUTOSTART=0
    JOTNAR_STATS=0
    JOTNAR_MDNS=0

    [ -f /etc/jotnar/node.conf ] && . /etc/jotnar/node.conf
    if [ -f "$JOTNAR_DATA_DIR/etc/node.conf" ]; then
        . "$JOTNAR_DATA_DIR/etc/node.conf"
    fi

    if [ -z "$JOTNAR_TOKEN" ] && [ -r "$JOTNAR_TOKEN_FILE" ]; then
        JOTNAR_TOKEN=$(cat "$JOTNAR_TOKEN_FILE" 2>/dev/null || true)
    fi

    # Accept both the old "host:port" bind form and host + port fields.
    case "$JOTNAR_BIND" in
        *:*) ;;
        *) JOTNAR_BIND="$JOTNAR_BIND:$JOTNAR_PORT" ;;
    esac

    export JOTNAR_HOSTNAME JOTNAR_BIND JOTNAR_PORT JOTNAR_ALLOW
    export JOTNAR_MAX_FRAME JOTNAR_TOKEN JOTNAR_TOKEN_FILE
    export JOTNAR_DATA_DIR JOTNAR_LOG_DIR JOTNAR_CONSOLE
    export JOTNAR_SSH JOTNAR_SSH_PORT JOTNAR_SUPERVISOR JOTNAR_STATS JOTNAR_MDNS
    export JOTNAR_OSD JOTNAR_MODULE_SOCKET
    export JOTNAR_CONSOLE_AUTOSTART
}
