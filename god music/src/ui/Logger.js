/**
 * Logger - Activity log system
 */

export class Logger {
    constructor(containerElement) {
        this.container = containerElement;
        this.maxEntries = 150;
    }

    /**
     * Log message
     */
    log(message, type = 'info') {
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        
        this.container.appendChild(entry);
        this.container.scrollTop = this.container.scrollHeight;

        // Limit entries
        while (this.container.children.length > this.maxEntries) {
            this.container.removeChild(this.container.firstChild);
        }
    }

    /**
     * Log info
     */
    info(message) {
        this.log(message, 'info');
    }

    /**
     * Log success
     */
    success(message) {
        this.log(message, 'success');
    }

    /**
     * Log warning
     */
    warning(message) {
        this.log(message, 'warning');
    }

    /**
     * Log error
     */
    error(message) {
        this.log(message, 'error');
    }

    /**
     * Clear log
     */
    clear() {
        this.container.innerHTML = '';
    }
}

