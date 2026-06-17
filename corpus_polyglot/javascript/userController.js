// Demo JavaScript module: a user controller with CRUD-style methods.
// Демонстрирует извлечение класса и его методов (Class.method) tree-sitter'ом.

const { hashPassword } = require("./auth");

class UserController {
  constructor(userRepository) {
    this.users = userRepository;
  }

  /**
   * Create a new user, hashing the password before persisting.
   */
  async createUser({ email, password, isActive = true }) {
    const passwordHash = await hashPassword(password);
    return this.users.insert({ email, passwordHash, isActive });
  }

  /**
   * Fetch a single user by id, or throw if it does not exist.
   */
  async getUserById(userId) {
    const user = await this.users.findById(userId);
    if (!user) {
      const error = new Error(`User ${userId} not found`);
      error.status = 404;
      throw error;
    }
    return user;
  }

  /**
   * Return a paginated list of users.
   */
  async listUsers({ skip = 0, limit = 25 } = {}) {
    return this.users.findMany({ skip, limit });
  }

  /**
   * Soft-delete a user by marking it inactive.
   */
  async deactivateUser(userId) {
    return this.users.update(userId, { isActive: false });
  }
}

module.exports = { UserController };
